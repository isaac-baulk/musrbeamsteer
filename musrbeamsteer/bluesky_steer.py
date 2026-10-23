from typing import Sequence, Optional
from gp_kg_minimizer import Observation
import os
import math
import lmfit
import scipp as sc
from collections.abc import Generator

import bluesky.plans as bp
import bluesky.plan_stubs as bps
import matplotlib
import matplotlib.pyplot as plt
from bluesky.utils import Msg
from ophyd_async.plan_stubs import ensure_connected
from ophyd_async.core import Device

from ibex_bluesky_core.callbacks import ISISCallbacks
from ibex_bluesky_core.utils import get_pv_prefix
from ibex_bluesky_core.devices.block import BlockWriteConfig, block_rw_rbv
from ibex_bluesky_core.devices.simpledae import (SimpleDae,
    RunPerPointController,
    PeriodPerPointController,
    PeriodGoodFramesNormalizer,
    MEventsWaiter, 
    PeriodGoodFramesWaiter,
    Reducer
)
from ibex_bluesky_core.devices.muon import MuonAsymmetryReducer, damped_oscillator, double_damped_oscillator
from ibex_bluesky_core.run_engine import get_run_engine

from ibex_bluesky_core.fitting import Gaussian
import matplotlib.pyplot as plt
from ibex_bluesky_core.plan_stubs import call_sync

import numpy as np

class GaussianProcessOptimizer:
    def __init__(
        self, 
        bounds: list[tuple[float, float]], 
        initial_coordinates: list[tuple[float, ...]],
        initial_repeats: int,
        accuracy: float,
        max_measurements: int,
        minimum_step_sizes: list[float]
    ):
        self.bounds = bounds
        self.accuracy = accuracy
        self.max_measurements = max_measurements
        self.minimum_step_sizes = minimum_step_sizes
        
        # State tracking
        self.observations: list[Observation] = []
        self.converged = False
        self.final_result = None
        self.estimated_noise_std = None
        
        # Pre-build the initial queue
        self._initial_queue = []
        for index, coord in enumerate(initial_coordinates):
            repeats = initial_repeats if index == 0 else 1
            for _ in range(repeats):
                self._initial_queue.append(coord)

    def ask(self) -> tuple[float, ...]:
        """Request the next coordinate to measure."""
        if self.converged or len(self.observations) >= self.max_measurements:
            self.converged = True
            return None # Or raise an exception to stop the Bluesky plan

        # 1. Initial Design Phase
        if self._initial_queue:
            return self._initial_queue.pop(0)
            
        # 2. Active Optimization Phase
        # Extract X and Y arrays from self.observations
        # Calculate your covariance matrix, noise estimate, and GP posterior here
        # (This is exactly the math currently sitting inside your search_minimum loop)
        
        # Run your gradient ascent to find the coordinate with the maximum KGCP
        best_candidate, max_kg, estimated_noise = self._maximize_knowledge_gradient()
        self.estimated_noise_std = estimated_noise

        # 3. Check stopping condition
        if max_kg < self.accuracy:
            self.converged = True
            # Calculate the final optimal coordinate using the posterior mean
            self.final_result = self._estimate_final_minimum()
            return None
            
        return best_candidate

    def tell(self, coordinate: tuple[float, ...], metric_value: float, run_number: str) -> None:
        """Provide the optimizer with the result of a measurement."""
        if self.converged:
            return
            
        obs = Observation(
            coordinate=coordinate,
            metric=metric_value,
            measurement={"run_number": run_number}
        )
        self.observations.append(obs)



def gp_adaptive_plan(optimizer: GaussianProcessOptimizer, steering_motors, detector, save_run=True, frames=200) -> Generator[Msg, None, None]:
    
    prefix = get_pv_prefix() #get prefix for current instrument
    controller = PeriodPerPointController(save_run=save_run) #continous run controller for DAE
    waiter = MEventsWaiter(target_mevents=MEVENTS_PER_MEASUREMENT)
        
    parameters = lmfit.Parameters()
    parameters.add("B", 0)
    parameters.add("A_0", 0.25, min=0)
    parameters.add("omega_0", (0.26 * 2*math.pi)/1000.)
    parameters.add("phi_0", 3.14)
    parameters.add("lambda_0", 0, vary=False)
    
    time_bins = sc.arange(dim="tof", unit=sc.units.ns, start=1952, stop=16000, step=32, dtype="float64")
   
    reducer = MuonAsymmetryReducer(
        prefix=prefix,
        forward_detectors=np.array([i for i in range(1, 96 + 1)]),
        backward_detectors=np.array([i for i in range(97, 192 + 1)]),
        alpha=1.0,
        model=lmfit.Model(damped_oscillator),
        fit_parameters=parameters,
        time_bin_edges=time_bins,
    )

    dae = SimpleDae(
        prefix=prefix,
        controller=controller,
        waiter=waiter,
        reducer=reducer,
    )
    while not optimizer.converged:
        next_coord = optimizer.ask()
        
        # The optimizer returns None when the KG threshold is met or max measurements hit
        if next_coord is None:
            break 
            
        # Move hardware
        yield from bps.mv(
            steering_motors[0], next_coord[0],
            steering_motors[1], next_coord[1]
        )
        
        # Trigger, wait, and read
        reading = yield from bps.trigger_and_read([detector])
        
        # Extract values
        metric_val = reading[detector.name]['value']
        run_num = reading[detector.name].get('run_number', 'unknown')
        
        # Feed back to optimizer
        optimizer.tell(next_coord, metric_val, run_num)

"""Demonstration plan showing basic bluesky functionality."""

default_prefix = get_pv_prefix()


def asymmetry_magnet_plan(
    magnet_device,
    start,
    stop,
    num,
    frames=500,
    save_run=True,
    prefix = default_prefix,
) -> Generator[Msg, None, None]:
    """Scan a DAE against a magnet."""
    
    plt.close("all")
    plt.show()

    controller = RunPerPointController(save_run=save_run)
    waiter = GoodFramesWaiter(frames)
        
    parameters = lmfit.Parameters()
    parameters.add("B", 0)
    parameters.add("A_0", 0.25, min=0)
    parameters.add("omega_0", (0.26 * 2*math.pi)/1000.)
    parameters.add("phi_0", 3.14)
    parameters.add("lambda_0", 0, vary=False)
    
    time_bins = sc.arange(dim="tof", unit=sc.units.ns, start=1952, stop=16000, step=32, dtype="float64")
   
    reducer = MuonAsymmetryReducer(
        prefix=prefix,
        forward_detectors=np.array([i for i in range(1, 96 + 1)]),
        backward_detectors=np.array([i for i in range(97, 192 + 1)]),
        alpha=1.0,
        model=lmfit.Model(damped_oscillator),
        fit_parameters=parameters,
        time_bin_edges=time_bins,
    )

    dae = SimpleDae(
        prefix=prefix,
        controller=controller,
        waiter=waiter,
        reducer=reducer,
    )

    yield from ensure_connected(magnet_device, dae, force_reconnect=False)

    if periods:
        yield from bps.mv(dae.number_of_periods, num)
    else:
        yield from bps.mv(dae.number_of_periods, 1)

    icc = ISISCallbacks(
        y=reducer.A_0.name,
        x=magnet_device.name,
        fit=Gaussian().fit(),
        measured_fields=[
            reducer.A_0_err.name,
            reducer.B.name, reducer.B_err.name,
            reducer.omega_0.name, reducer.omega_0_err.name,
            reducer.phi_0.name, reducer.phi_0_err.name,
            reducer.r_squared.name],
    )

    @icc
    def _inner() -> Generator[Msg, None, None]:
        yield from bp.scan([dae], magnet_device, start, stop, num=num)

    yield from _inner()
    
    # if icc.live_fit.result is not None:
    #     print(icc.live_fit.result.fit_report())
    #     fitted_value = icc.live_fit.result.params["x0"].value
    #     
    #     if not (args[1] <= fitted_value <= args[2]):
    #         print(f"Fitted value was outside range, using max {icc.peak_stats['max']}")
    #         fitted_value = icc.peak_stats['max'][0]
    #     
    #     if not confident:
    #         ans = yield from call_sync(input, f"set {args[0].name} to {fitted_value}? y/n")
    #     else:
    #         ans = "y"
    #     if ans.lower().startswith("y"):
    #         fitted_fraction = (fitted_value - args[1]) / (args[2] - args[1])
    #         for device, start, stop in batched(args, n=3):
    #             value = start + fitted_fraction * (stop - start)
    #             print(f"Setting {device.name} to fitted peak {value}")
    #             yield from bps.mv(device, value)
    # else:
    #     print("Fit failed to converge")

    return icc.live_fit.result