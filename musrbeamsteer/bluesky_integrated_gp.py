"""
Native Bluesky automated beam-steering optimization script.
Replaces legacy genie_python and mantid file-polling with Ophyd devices and an adaptive RunEngine plan.
"""

import asyncio
import time
import numpy as np
import scipp as sc
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import abc
from typing import Any, Mapping, Sequence

from bluesky import RunEngine
import bluesky.plan_stubs as bps
from bluesky.protocols import Status

from ophyd_async.core import StandardReadable, soft_signal_r_and_setter, Device, AsyncStatus

from scipy.stats import qmc

from ibex_bluesky_core.devices.simpledae import (SimpleDae,
    RunPerPointController,
    PeriodPerPointController,
    PeriodGoodFramesNormalizer,
    MEventsWaiter,
    Reducer, 
    PeriodGoodFramesWaiter
)
from ibex_bluesky_core.devices.block import BlockMot
from ibex_bluesky_core.devices.dae import Dae, DaeSpectra
from ibex_bluesky_core.utils import calculate_polarisation

# Import your PCA logic and Observation dataclass
from PCA.data_handler import DataHandler
from PCA.pca_machine import PCAMachine

import math
import numpy as np
from typing import Sequence
from gp_kg_minimizer import (
    Observation,
    gp_kg_next_coordinate,
    _estimate_final_minimum,
    _apply_minimum_step_sizes,
    _validate_bounds,
    _validate_minimum_step_sizes,
    _validate_precisions,
    _round_coordinate,
    estimate_noise_variance
)

# ==========================================================================
# USER CONFIGURATION
# ==========================================================================
LIVE_PLOT = True
TEST_MODE = True
LONGITUDINAL_MODE = True
METRIC_FUNCTION = "PCA score"  # "Integrated asymmetry" or "PCA score"
MEVENTS_PER_MEASUREMENT = 1.0
SETTLE_SECONDS = 2.0
ACCURACY = 0.0001
INITIAL_REPEATS = 2
MAX_MEASUREMENTS = 30
MIN_PC1_SIGNAL_CAPTURE = 0.3

SCAN_PVS = [
    {"name": "hsm_steering", "pv": "IN:MUSR:KEPCO_04:CURRENT:SP", "bounds": (-0.5, 0.5), "precision": 2, "minimum_step_size": 0.01, "optimum_guess": 0.0},
    {"name": "vsm_steering", "pv": "IN:MUSR:KEPCO_05:CURRENT:SP", "bounds": (-1.5, 1.5), "precision": 2, "minimum_step_size": 0.01, "optimum_guess": 0.0},
]
NUM_INITIAL_COORDINATES = len(SCAN_PVS) * 3

SPECTRA_CFG = {
    True: {"forward": list(range(33, 65)), "backward": list(range(1, 33))},
    False: {"forward": list(range(9, 17)) + list(range(57, 65)), "backward": list(range(25, 33)) + list(range(41, 49))}
}[LONGITUDINAL_MODE]

# ==========================================================================
# LIVE PLOT
# ==========================================================================

class LiveOptimizerPlot:
    def __init__(self, bounds: list[tuple[float, float]], pv_names: list[str]):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.fig.canvas.toolbar = None
        self.bounds = bounds
        self.pv_names = pv_names

        # Pre-allocate the colorbar axis before the RunEngine starts
        # This prevents Tkinter from throwing thread errors during dynamic widget creation
        dummy_sc = self.ax.scatter([-999], [-999], c=[0], cmap='viridis', visible=False)
        self.cbar = self.fig.colorbar(dummy_sc, ax=self.ax)
        self.cbar.set_label('Metric Score')

    def update(self, observations: Sequence[Observation]) -> None:
        self.ax.clear()
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        x_vals, y_vals, metrics, indices = [], [], [], []
        for i, obs in enumerate(observations):
            if np.isfinite(obs.metric):
                x_vals.append(obs.coordinate[0])
                y_vals.append(obs.coordinate[1])
                metrics.append(obs.metric)
                indices.append(i + 1)
                
        if not x_vals:
            self.fig.canvas.flush_events()
            return

        min_x, max_x = self.bounds[0]
        min_y, max_y = self.bounds[1]
        self.ax.plot(
            [min_x, max_x, max_x, min_x, min_x], 
            [min_y, min_y, max_y, max_y, min_y], 
            'k--', linewidth=2
        )

        for i in range(1, len(x_vals)):
            self.ax.annotate('', xy=(x_vals[i], y_vals[i]), xytext=(x_vals[i-1], y_vals[i-1]),
                             arrowprops=dict(arrowstyle='->', color='gray', shrinkA=6, shrinkB=6))

        sc = self.ax.scatter(x_vals, y_vals, c=metrics, cmap='viridis', s=80, edgecolors='k', zorder=3)

        # Update the existing colorbar mapping instead of creating a new axis
        self.cbar.update_normal(sc)

        for i, idx in enumerate(indices):
            self.ax.text(x_vals[i] + 0.015, y_vals[i] + 0.015, str(idx), fontsize=9)

        min_idx = np.argmin(metrics)
        self.ax.plot(x_vals[min_idx], y_vals[min_idx], 'o', markersize=16, markerfacecolor='none', markeredgecolor='k', markeredgewidth=2, zorder=2)
        
        self.ax.set_xlabel(f"{self.pv_names[0]} steering current")
        self.ax.set_ylabel(f"{self.pv_names[1]} steering current")
        self.ax.set_title('Live Metric Optimization')
        self.ax.set_xlim(min_x - 0.1, max_x + 0.1)
        self.ax.set_ylim(min_y - 0.1, max_y + 0.1)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

# ==========================================================================
# OPHYD HARDWARE DEVICES
# ==========================================================================

class SimulatedDAE(Device):
    """A mock detector generating synthetic asymmetry data asynchronously."""
    
    def __init__(self, name, motors):
        self.motors = motors
        # Internal state to hold the generated arrays
        self._time_array = np.array([])
        self._asym_array = np.array([])
        self._error_array = np.array([])
        super().__init__(name=name)

    @AsyncStatus.wrap
    async def trigger(self):
        """Simulates starting a run and generating synthetic arrays."""
        # Read the current position of the simulated motors
        
        x = await self.motors[0].user_readback.get_value()
        y = await self.motors[1].user_readback.get_value()

        distance = np.sqrt(x**2 + y**2)
        t_arr = np.linspace(0.1, 15.0, 100)
        
        relaxation_rate = max(0.0, 1.0 - distance)
        clean_asym = 0.25 * np.exp(-relaxation_rate * t_arr)
        
        noise_level = 0.01 / np.sqrt(max(1.0, MEVENTS_PER_MEASUREMENT))
        self._asym_array = clean_asym + np.random.normal(0, noise_level, len(t_arr))
        self._time_array = t_arr
        self._error_array = np.full(len(t_arr), noise_level)

    async def read(self):
        """Returns the exact dictionary structure the Bluesky plan expects."""
        ts = time.time()
        return {
            f"{self.name}-reducer-time_array": {"value": self._time_array, "timestamp": ts},
            f"{self.name}-reducer-asym_array": {"value": self._asym_array, "timestamp": ts},
            f"{self.name}-reducer-error_array": {"value": self._error_array, "timestamp": ts},
        }

    async def describe(self):
        """Bluesky requires a description of the data types returned by read()."""
        return {
            f"{self.name}-reducer-time_array": {"source": "SIM", "dtype": "array", "shape": [100]},
            f"{self.name}-reducer-asym_array": {"source": "SIM", "dtype": "array", "shape": [100]},
            f"{self.name}-reducer-error_array": {"source": "SIM", "dtype": "array", "shape": [100]},
        }
        
    async def read_configuration(self):
        return {}
        
    async def describe_configuration(self):
        return {}

class RawMuonAsymmetryReducer(Reducer, StandardReadable):
    """DAE reducer that bypasses fitting to expose raw asymmetry arrays."""
    
    def __init__(self, prefix: str, forward_detectors: np.ndarray, backward_detectors: np.ndarray, alpha: float = 1.0, time_bin_edges: sc.Variable = None):
        self._forward_detectors = forward_detectors
        self._backward_detectors = backward_detectors
        self._alpha = alpha
        self._time_bin_edges = time_bin_edges
        self._first_det = DaeSpectra(dae_prefix=prefix + "DAE:", spectra=int(forward_detectors[0]), period=0)
        
        self.time_array, self._time_setter = soft_signal_r_and_setter(np.ndarray, np.array([]))
        self.asym_array, self._asym_setter = soft_signal_r_and_setter(np.ndarray, np.array([]))
        self.error_array, self._error_setter = soft_signal_r_and_setter(np.ndarray, np.array([]))

        super().__init__(name="")

    def _rebin_and_sum(self, counts: np.ndarray, time_coord: sc.Variable) -> sc.DataArray:
        da = sc.DataArray(
            data=sc.array(dims=["spec", "tof"], values=counts, variances=counts, unit=sc.units.counts, dtype="float64"),
            coords={"tof": time_coord},
        )
        da = da.sum(dim="spec")
        if self._time_bin_edges is not None:
            da = da.rebin({"tof": self._time_bin_edges})
        return da

    def _calculate_asymmetry(self, current_period_data: np.ndarray, first_spec_dataarray: sc.DataArray) -> sc.DataArray:
        forward = self._rebin_and_sum(current_period_data[self._forward_detectors, :], first_spec_dataarray.coords["tof"])
        backward = self._rebin_and_sum(current_period_data[self._backward_detectors, :], first_spec_dataarray.coords["tof"])
        forward.variances += 0.5
        backward.variances += 0.5
        return calculate_polarisation(forward, backward, self._alpha)

    async def reduce_data(self, dae: Dae) -> None:
        current_period_data, first_spec_dataarray = await asyncio.gather(
            dae.trigger_and_get_specdata(),
            self._first_det.read_spectrum_dataarray(),
        )
        asymmetry = self._calculate_asymmetry(current_period_data, first_spec_dataarray)
        tof = asymmetry.coords["tof"].values
        if len(tof) == len(asymmetry.values) + 1:
            tof = (tof[:-1] + tof[1:]) / 2
            
        self._time_setter(tof)
        self._asym_setter(asymmetry.values)
        self._error_setter(asymmetry.variances ** 0.5)

    def additional_readable_signals(self, dae: Dae) -> list:
        return [self.time_array, self.asym_array, self.error_array]


# ==========================================================================
# METRIC EXTRACTORS
# ==========================================================================

class BaseMetricExtractor(abc.ABC):
    def __call__(
        self, 
        measurements: Sequence[Mapping[str, Any]], 
        asymm_array: np.ndarray, 
        pvs: list[dict[str, float | int]],
        first_set: bool = False
    ) -> list[float]:
        self.setup_calc(asymm_array, pvs)
        if first_set:
            self.verify_statistics()
        return self.calculate_metrics(measurements, asymm_array, pvs)

    @abc.abstractmethod
    def setup_calc(self, asymm_array: np.ndarray, pvs: list[dict[str, float | int]]) -> None: pass

    @abc.abstractmethod
    def verify_statistics(self) -> None: pass

    @abc.abstractmethod
    def calculate_metrics(self, measurements: Sequence[Mapping[str, Any]], asymm_array: np.ndarray, pvs: list[dict[str, float | int]]) -> list[float]: pass


class IntegratedAsymmetryExtractor(BaseMetricExtractor):
    def setup_calc(self, asymm_array: np.ndarray, pvs: list[dict[str, float | int]]) -> None: pass
    def verify_statistics(self) -> None: pass

    def calculate_metrics(self, measurements: Sequence[Mapping[str, Any]], asymm_array: np.ndarray, pvs: list[dict[str, float | int]]) -> list[float]:
        score_map = {}
        for j, pv_data in enumerate(pvs):
            run_num = str(pv_data["run"])
            asymmetry = asymm_array[:, j, 1]
            errors = asymm_array[:, j, 2]

            valid = np.isfinite(asymmetry) & np.isfinite(errors) & (errors > 0)
            if not np.any(valid):
                raise ValueError(f"Run {run_num} contains no finite bins with positive errors.")

            weights = 1.0 / np.square(errors[valid])
            weighted_asymmetry = np.sum(asymmetry[valid] * weights) / np.sum(weights)
            score_map[run_num] = float(weighted_asymmetry)
            
        metrics = []
        for m in measurements:
            run_str = str(m["run_number"]).lstrip("0") or "0"
            if run_str in score_map:
                metrics.append(score_map[run_str])
            else:
                raise ValueError(f"Run {run_str} requested by optimizer but not found in output.")
        return metrics


class PCAMetricExtractor(BaseMetricExtractor):
    def setup_calc(self, asymm_array: np.ndarray, pvs: list[dict[str, float | int]]) -> None:
        self.pvs = pvs
        self.dh = DataHandler()
        self.dh.load_batch_from_array(asymm_array, name="live_optimisation_batch")
        self.dh.prepare_XYE_PCA()
        self.pca_machine = PCAMachine(self.dh)
        self.pca_machine.perform_pca()

    def verify_statistics(self) -> None:     
        self.pca_machine.print_pca_representation()
        singular_values = np.asarray(self.pca_machine.pc_sing[0], dtype=float)
        if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
            raise ValueError("Cannot evaluate PCA signal capture: invalid singular values.")

        pc1_signal_capture = float(np.square(singular_values[0]) / np.sum(np.square(singular_values)))
        print(f"First-set PCA quality check: PC1 captures {pc1_signal_capture * 100:.1f}% of signal.")
        
        if pc1_signal_capture < MIN_PC1_SIGNAL_CAPTURE:
            raise SystemExit(f"Aborting optimization: PC1 capture too low ({pc1_signal_capture * 100:.1f}%).")

    def calculate_metrics(self, measurements: Sequence[Mapping[str, Any]], asymm_array: np.ndarray, pvs: list[dict[str, float | int]]) -> list[float]:
        pc1_curve = self.pca_machine.pc_curves[0][:, 0]
        pc1_errors = np.asarray(self.dh.prepared_data[0][2], dtype=float)
        valid = np.isfinite(pc1_curve) & np.isfinite(pc1_errors) & (pc1_errors > 0)
        
        weights = 1.0 / np.square(pc1_errors[valid])
        if np.sum(pc1_curve[valid] * weights) < 0:
            self.pca_machine.pc_curves[0][:, 0] *= -1
            self.pca_machine.pc_scores[0][0, :] *= -1

        pc1_scores = self.pca_machine.pc_scores[0][0, :].copy()
        score_map = {str(pv_data["run"]): score for pv_data, score in zip(self.pvs, pc1_scores)}
        
        metrics = []
        for m in measurements:
            run_str = str(m["run_number"]).lstrip("0") or "0"
            if run_str in score_map:
                metrics.append(score_map[run_str])
        return metrics

class GaussianProcessOptimizer:
    """State machine wrapping the GP Knowledge-Gradient search."""
    
    def __init__(
        self, 
        bounds: Sequence[Sequence[float]], 
        initial_coordinates: list[tuple[float, ...]], 
        initial_repeats: int, 
        accuracy: float, 
        max_measurements: int,
        minimum_step_sizes: Sequence[float] = None,
        precisions: Sequence[int] = None,
        random_state: int = 12345
    ):
        # 1. Validate inputs using the existing minimizer logic
        self.bounds = _validate_bounds(bounds)
        dimension = self.bounds.shape[0]
        self.minimum_steps = _validate_minimum_step_sizes(minimum_step_sizes, dimension)
        self.precisions = _validate_precisions(precisions, dimension)
        
        self.accuracy = accuracy
        self.max_measurements = max_measurements
        self.random_state = random_state
        
        # State tracking
        self.observations: list[Observation] = []
        self.converged = False
        self.final_result = None
        self.estimated_noise_std = None
        self.kg = None
        
        # 2. Pre-build the initial Latin Hypercube queue[cite: 5]
        self._initial_queue = []
        for index, coord in enumerate(initial_coordinates):
            rounded_coord = _round_coordinate(coord, self.precisions)
            # Only repeat the first coordinate to estimate noise variance[cite: 5]
            repeats = initial_repeats if index == 0 else 1
            for _ in range(repeats):
                self._initial_queue.append(rounded_coord)

    def ask(self) -> tuple[float, ...] | None:
        """Request the next optimal coordinate to measure."""
        if self.converged or len(self.observations) >= self.max_measurements:
            self.converged = True
            return None

        # 1. Exhaust the initial Latin Hypercube queue first
        if self._initial_queue:
            return tuple(self._initial_queue.pop(0))
            
        # 2. Calculate the next coordinate using the approximate correlated Knowledge Gradient[cite: 5]
        # We advance the random state based on observation count to ensure varied Sobol sampling[cite: 5]
        raw_coordinate, max_kg, noise_std = gp_kg_next_coordinate(
            observations=self.observations,
            bounds=self.bounds,
            random_state=self.random_state + len(self.observations)
        )
        
        self.kg = max_kg
        self.estimated_noise_std = noise_std
        
        # 3. Apply minimum movement and precision constraints[cite: 5]
        adjusted_coordinate = _apply_minimum_step_sizes(
            coordinate=raw_coordinate,
            previous_coordinate=self.observations[-1].coordinate,
            minimum_step_sizes=self.minimum_steps,
            precisions=self.precisions,
            bounds=self.bounds
        )

        # 4. Check the stopping condition against expected one-step improvement[cite: 5]
        if self.kg < self.accuracy:
            self.converged = True
            
            # Estimate the final minimum from the fitted posterior mean[cite: 5]
            final_coord, final_val = _estimate_final_minimum(
                observations=self.observations,
                bounds=self.bounds,
                random_state=self.random_state + 99999
            )
            self.final_result = tuple(_round_coordinate(final_coord, self.precisions))
            return None
            
        return tuple(adjusted_coordinate)

    def tell(self, coordinate: tuple[float, ...], metric_value: float, run_number: str) -> None:
        """Provide the optimizer with the latest experimental result."""
        if self.converged:
            return
            
        self.observations.append(
            Observation(
                coordinate=np.asarray(coordinate, dtype=float), 
                metric=float(metric_value), 
                measurement={"run_number": run_number}
            )
        )

# ==========================================================================
# BLUESKY ADAPTIVE PLAN
# ==========================================================================

def gp_adaptive_plan(optimizer, steering_motors, dae_device, metric_extractor, live_plot):
    historical_asymm_list = []
    historical_pvs = []

    yield from bps.open_run()

    while not optimizer.converged:
        next_coord = optimizer.ask()
        if next_coord is None:
            break 
            
        yield from bps.mv(
            steering_motors[0], next_coord[0],
            steering_motors[1], next_coord[1]
        )
        yield from bps.sleep(SETTLE_SECONDS)
        
        reading = yield from bps.trigger_and_read([dae_device], name='optimizer_step')
        
        t_arr = reading[f"{dae_device.name}-reducer-time_array"]['value']
        a_arr = reading[f"{dae_device.name}-reducer-asym_array"]['value']
        e_arr = reading[f"{dae_device.name}-reducer-error_array"]['value']
        
        current_data = np.column_stack((t_arr, a_arr, e_arr))
        historical_asymm_list.append(current_data)
        asymm_array = np.stack(historical_asymm_list, axis=1)
        
        current_pv_dict = {"run": len(historical_pvs) + 1, "hsm_steering": next_coord[0], "vsm_steering": next_coord[1]}
        historical_pvs.append(current_pv_dict)
        
        is_first_set = (len(historical_pvs) == NUM_INITIAL_COORDINATES)
        dummy_measurements = [{"run_number": pv["run"]} for pv in historical_pvs]
        
        metrics = metric_extractor(dummy_measurements, asymm_array, historical_pvs, first_set=is_first_set)
        
        optimizer.tell(next_coord, metrics[-1], run_number=str(current_pv_dict["run"]))
        
        if live_plot is not None:
            live_plot.update(optimizer.observations)

    yield from bps.close_run()


# ==========================================================================
# EXECUTION
# ==========================================================================

def main():
    # 1. Setup RunEngine
    RE = RunEngine({})

    # 2. Setup Initial Coordinates
    lower_bounds = np.array([item["bounds"][0] for item in SCAN_PVS])
    upper_bounds = np.array([item["bounds"][1] for item in SCAN_PVS])
    lhs_sampler = qmc.LatinHypercube(d=len(SCAN_PVS))
    initial_coordinates = [tuple(float(v) for v in row) for row in qmc.scale(lhs_sampler.random(n=NUM_INITIAL_COORDINATES), lower_bounds, upper_bounds)]

    # 3. Instantiate Components
    optimizer = GaussianProcessOptimizer(
        bounds=[item["bounds"] for item in SCAN_PVS],
        initial_coordinates=initial_coordinates,
        initial_repeats=INITIAL_REPEATS,
        accuracy=ACCURACY,
        max_measurements=MAX_MEASUREMENTS
    )

    extractor_class = {"Integrated asymmetry": IntegratedAsymmetryExtractor, "PCA score": PCAMetricExtractor}[METRIC_FUNCTION]
    metric_extractor = extractor_class()

    live_plot = LiveOptimizerPlot([item["bounds"] for item in SCAN_PVS], [item["name"] for item in SCAN_PVS]) if LIVE_PLOT else None


    # block_mot provides the Movable interface required by bps.mv()
    hsm = BlockMot(prefix="IN:MUSR:", block_name="hsm_steering")
    vsm = BlockMot(prefix="IN:MUSR:", block_name="vsm_steering")
    hsm.set_name("hsm_steering")
    vsm.set_name("vsm_steering")

    # 4. Create the DAE
    if TEST_MODE:
        # Use custom simulator that bypasses EPICS state machine logic
        musr_dae = SimulatedDAE(name="musr_dae", motors=[hsm, vsm])
    else:
        # Use real IBEX DAE
        musr_dae = SimpleDae(
            prefix="IN:MUSR:",
            controller=RunPerPointController(save_run=True),
            waiter=MEventsWaiter(MEVENTS_PER_MEASUREMENT),
            reducer=RawMuonAsymmetryReducer(
                prefix="IN:MUSR:",
                forward_detectors=np.array(SPECTRA_CFG["forward"]),
                backward_detectors=np.array(SPECTRA_CFG["backward"]),
                alpha=1.0
            )
        )
        musr_dae.set_name("musr_dae")

    # 5. Connect concurrently and explicitly
    async def connect_hardware():
        await asyncio.gather(
            hsm.connect(mock=TEST_MODE),
            vsm.connect(mock=TEST_MODE),
            musr_dae.connect(mock=TEST_MODE)
        )
    asyncio.run(connect_hardware())

    # 4. Run the Plan
    RE(gp_adaptive_plan(optimizer, [hsm, vsm], musr_dae, metric_extractor, live_plot))

    print(f"Optimizer Result: {optimizer.final_result}. Noise Estimate: {optimizer.estimated_noise_std}. Measurements: {len(optimizer.observations)}.")

    if live_plot is not None:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()