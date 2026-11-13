"""MUSR 2-D steering magnet test, with no muon beam / no DAE required.

Edit the CONFIGURATION section, especially the actual MUSR steering-magnet
EPICS setpoint/readback PVs, readback tolerances and approved current bounds.
The old ARGUS magnet PV definitions MUST NOT be reused here.

``USE_SYNTHETIC_DATA=True``: use simulated data and do not connect to the DAE.
``USE_SYNTHETIC_MOTORS=True``: use in-memory motors and do not connect to EPICS.
The switches are independent, so synthetic data can be tested with real motors
or live data can be tested with synthetic motors.

The synthetic signal has a position-dependent muon rate (10% at the
parameter-space boundary) and constant TRUE A(0). Both live-histogram
INTEGRATED ASYMMETRY and PCA metrics can locate the Kubo--Toyabe centre.
"""
from __future__ import annotations

from functools import partial
import ast
import re
import pandas as pd
import numpy as np
from pathlib import Path

from blueskysteercore.acquisition import make_live_ibex_dae_acquire, make_synthetic_acquire
from blueskysteercore.benchmarking import BenchmarkRecorder, duration_as_dict
from blueskysteercore.bluesky_integration import minimize_plan
from blueskysteercore.gp_kg_minimizer import gaussian_initial_coordinates
from blueskysteercore.live_monitor import live_metric_monitor
from blueskysteercore.isis_dae import AcquisitionDuration, make_live_dae
from blueskysteercore.musr_metrics import (
    asym_diff_count_rate_metric,
    integrated_asymmetry_metric,
    pca_asymmetry_metric,
    asym_squared_count_rate_metric
)

from blueskysteercore.positioners import make_setpoint_readback_axis
from blueskysteercore.synthetic import SyntheticMotor, generate_synthetic_musr_data


# ========================= CONFIGURATION =========================
USE_SYNTHETIC_DATA = True  # False: use the LIVE MUSR DAE
USE_SYNTHETIC_MOTORS = True  # False: use the LIVE MUSR steering magnets
MAX_PERIODS = 60 #limited by memory card of instrument
PERIOD_PER_POINT = True  # False: one DAE run per point; True: all points in one DAE run
DATA_READ_TYPE = 'snapshot' # RunPerPoint: 'nexus' reads saved files; 'snapshot' reads DAE memory
INCOMING_DIR = Path("//musr/Data") #configure if reading nexus files
benchmark_acq = False  # Enable DAE timing and configuration logging.

INSTRUMENT = "MUSR_LONGITUDINAL" #other options: MUSR_TRANSVERSE, EMU, HIFI

PV_DF = pd.DataFrame(
    columns=["name", "type", "setpoint_pv", "readback_pv", "tolerance", "initial_guess", "safe_bounds", "settle_seconds", "minimum_step_size", "precision", "metric"],
    data=[
        ["musr_steering_1", "steering", "IN:MUSR:KEPCO_04:CURRENT:SP", "IN:MUSR:KEPCO_04:CURRENT", 0.01, 0.0, (-0.5, 0.5), 1.0, 0.01, 3, "integrated_asymmetry"],
        ["musr_steering_2", "steering", "IN:MUSR:KEPCO_05:CURRENT:SP", "IN:MUSR:KEPCO_05:CURRENT", 0.01, 0.0,(-1.5, 1.5), 1.0, 0.01, 3, "integrated_asymmetry"],
        ["musr_focusing_1", "focusing", "IN:MUONFE:GENESYS_03:1:CURR:SP", "IN:MUONFE:GENESYS_03:1:CURR", 0.01, 0.0, (-2.0, 2.0), 1.0, 0.01, 3, "asym_diff_count_rate"],
        ["musr_focusing_2", "focusing", "IN:MUONFE:GENESYS_03:2:CURR:SP", "IN:MUONFE:GENESYS_03:2:CURR", 0.01, 0.0, (-2.0, 2.0), 1.0, 0.01, 3, "asym_diff_count_rate"],
        ["musr_slits", "collimation", "IN:MUSR:MUONJAWS_01:POSN:SP", "IN:MUSR:MUONJAWS_01:POSN", 0.01, 12.0, (6.0, 20.0), 1.0, 0.01, 3, "asym_diff_count_rate"],
    ]
)

# Ordered optimisation stages. Repeated groups are intentional.
STEER_ORDER_ACCURACIES = (
    ("steering", 0.1),
    ("focusing", 0.001),
    ("steering", 0.001),
    ("collimation", 0.001),
)

MOVE_TIMEOUT_SECONDS = 30.0

SYNTHETIC_MOTORS = tuple(
    SyntheticMotor(
        name=row["name"],
        setpoint_pv=row["setpoint_pv"],
        readback_pv=row["readback_pv"],
        safe_bounds=row["safe_bounds"],
        initial_value=row["initial_guess"],
        readback_tolerance=row["tolerance"],
        timeout=MOVE_TIMEOUT_SECONDS,
    )
    for _, row in PV_DF.iterrows()
)

INITIAL_EDGE_MARGIN_FRACTION = 0.10  # keep initial points 10% from each edge
# Each dimension is sampled with sigma=distance to the nearest bound / 3.
# Initial draws are rejected, not clipped, when outside the interior margin.

# Use the SAME duration for synthetic and real modes.
# DURATION = AcquisitionDuration.good_frames(500)
DURATION = AcquisitionDuration.events(1_000_000)
# DURATION = AcquisitionDuration.seconds(10.0)

# N-dimensional simulated sample centre / Gaussian spatial curvature.
# These describe the synthetic beam, NOT the user's initial guess.
CENTER = [0.1, -0.3, 0.0, 0.0, 10.0]
CURVATURE = [3.0, 3.0, 1.0, 1.0, 0.001]  # inverse-square of PV units

# Static zero-field Gaussian Kubo--Toyabe parameters.
KT_DELTA_US_INV = 0.75       # KT width Delta in microseconds^-1
INITIAL_ASYMMETRY = 0.23    # same TRUE t=0 asymmetry at ALL coordinates
MUON_LIFETIME_US = 2.1969811
COUNTS_PER_GOOD_FRAME = 1000.0  # at the synthetic centre
EDGE_RATE_FRACTION = 0.10  # expected muons/frame at ANY boundary face
GOOD_FRAMES_PER_SECOND = 50.0  # illustrative for SYNTHETIC seconds/events
BIN_WIDTH_NS = 16.0
MAX_TIME_US = 32.0
N_TIME_BINS = round(MAX_TIME_US * 1000 / BIN_WIDTH_NS)  # 2000 bins, [0, 32 us)
RANDOM_SEED = 20261023

# Select the live-histogram metric without changing the optimiser/acquisition.
METRIC_TIME_WINDOW_NS = (0.0, 30000.0)  # synthetic t0=0; calibrate for live DAE
METRIC_REBIN = 1  # native 16 ns bins; set 8 for ~128 ns channels
METRIC_ALPHA = 1.0  # calibrate for real MUSR measurements
BACKGROUND_ASYMMETRY = 0.2  # calibrate for real MUSR measurements
MIN_BIN_COUNTS = 10.0

MAX_MEASUREMENTS = 50
INITIAL_REPEATS = 2
# 0: quiet, 1: measurements + best measured, 2: also show GP estimate/diagnostics.
VERBOSE = 2
MINIMUM_ESTIMATE_POINTS = 1024  # used only when VERBOSE=2 (no motor moves)

# Live plots: 2D GP-predicted metric + latest forward/backward asymmetry.
PLOT_LIVE = True
PLOT_AXES = None  # default (0, 1); e.g. (1, 2) in a 3D+ scan
PLOT_GRID_SIZE = 65  # 65 x 65 GP posterior-mean colour map
PLOT_MINIMUM_POINTS = 1024  # full N-D Sobol search for slice centre
PLOT_ASYMMETRY_REBIN = 8  # 8 x 16 ns -> 128 ns per plotted point
PLOT_ASYMMETRY_TIME_WINDOW_NS = (0.0, 32000.0)
# To save a PNG instead of opening a Qt window: set PLOT_LIVE=False and
# PLOT_IMAGE_PATH=r"C:\Users\musr\Documents\Isaac\musr_monitor.png".
PLOT_IMAGE_PATH = None
# ================================================================


def make_simulator():
    """Create the full-dimensional synthetic acquisition simulator."""
    return _full_simulator()


def _metric_for_group(metric_name: str):
    options = dict(
        alpha=METRIC_ALPHA,
        time_window_ns=METRIC_TIME_WINDOW_NS,
        rebin=METRIC_REBIN,
        min_bin_counts=MIN_BIN_COUNTS,
    )
    if metric_name == "integrated_asymmetry":
        return integrated_asymmetry_metric(**options)
    if metric_name == "pca":
        return pca_asymmetry_metric(**options)
    if metric_name == "asym_squared_count_rate":
        return asym_squared_count_rate_metric(**options)
    if metric_name == "asym_diff_count_rate":
        return asym_diff_count_rate_metric(
            **options, background_asym=BACKGROUND_ASYMMETRY
        )
    raise ValueError(f"Unsupported metric {metric_name!r}")


def _validate_pv_configuration() -> None:
    required = {
        "name", "type", "setpoint_pv", "readback_pv", "tolerance",
        "initial_guess", "safe_bounds", "settle_seconds",
        "minimum_step_size", "precision", "metric",
    }
    if set(PV_DF.columns) != required:
        raise ValueError(f"PV_DF columns must be exactly {sorted(required)}")
    if PV_DF.empty or PV_DF["name"].duplicated().any():
        raise ValueError("PV_DF must contain unique motor names")
    if not USE_SYNTHETIC_MOTORS and (
        PV_DF["setpoint_pv"].eq("").any() or PV_DF["readback_pv"].eq("").any()
    ):
        raise ValueError("Live mode requires setpoint and readback PVs for every PV_DF row")
    for _, row in PV_DF.iterrows():
        try:
            lower, upper = (float(value) for value in row.safe_bounds)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid safe_bounds for {row.name!r}") from None
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid safe_bounds for {row.name!r}")
        if not lower <= row.initial_guess <= upper:
            raise ValueError(f"initial_guess outside safe_bounds for {row.name!r}")
        if row.settle_seconds < 0 or row.tolerance <= 0:
            raise ValueError(f"Invalid timing/tolerance for {row.name!r}")
        if not isinstance(row.precision, (int, np.integer)) or row.precision < 0:
            raise ValueError(f"Invalid precision for {row.name!r}")
    known_groups = set(PV_DF["type"])
    for group, accuracy in STEER_ORDER_ACCURACIES:
        if group not in known_groups:
            raise ValueError(f"Optimisation group {group!r} has no PV_DF rows")
        if not np.isfinite(accuracy) or accuracy <= 0:
            raise ValueError(f"Invalid accuracy for {group!r}")


def _full_simulator():
    return partial(
        generate_synthetic_musr_data,
        center=CENTER,
        curvature=CURVATURE,
        bounds=[tuple(value) for value in PV_DF["safe_bounds"]],
        delta_us_inv=KT_DELTA_US_INV,
        initial_asymmetry=INITIAL_ASYMMETRY,
        duration=DURATION,
        muon_lifetime_us=MUON_LIFETIME_US,
        counts_per_frame=COUNTS_PER_GOOD_FRAME,
        edge_rate_fraction=EDGE_RATE_FRACTION,
        good_frames_per_second=GOOD_FRAMES_PER_SECOND,
        n_time_bins=N_TIME_BINS,
        max_time_us=MAX_TIME_US,
        rng=np.random.default_rng(RANDOM_SEED),
    )


def make_plan():
    """Return sequential optimisation plans for the configured PV groups."""
    _validate_pv_configuration()
    from ophyd_async.plan_stubs import ensure_connected
    import bluesky.plan_stubs as bps

    rows = list(PV_DF.itertuples(index=False))
    if USE_SYNTHETIC_MOTORS:
        motors = [
            SyntheticMotor(
                name=row.name,
                setpoint_pv=row.setpoint_pv,
                readback_pv=row.readback_pv,
                safe_bounds=tuple(row.safe_bounds),
                initial_value=float(row.initial_guess),
                readback_tolerance=float(row.tolerance),
                timeout=MOVE_TIMEOUT_SECONDS,
            ).build()
            for row in rows
        ]
    else:
        if any(not row.setpoint_pv or not row.readback_pv for row in rows):
            raise ValueError("Live mode requires setpoint and readback PVs for every PV_DF row")
        motors = [
            make_setpoint_readback_axis(
                name=row.name,
                setpoint_pv=row.setpoint_pv,
                readback_pv=row.readback_pv,
                tolerance=float(row.tolerance),
                timeout=MOVE_TIMEOUT_SECONDS,
                limits=tuple(row.safe_bounds),
            )
            for row in rows
        ]

    recorder = BenchmarkRecorder() if benchmark_acq else None
    if USE_SYNTHETIC_DATA:
        acquire_full = make_synthetic_acquire(_full_simulator())
        dae = None
    else:
        dae = make_live_dae(
            duration=DURATION,
            instrument=INSTRUMENT,
            period_per_point=PERIOD_PER_POINT,
            data_read_type=DATA_READ_TYPE,
            nexus_directory=INCOMING_DIR,
            read_time_axis=True,
            benchmark_recorder=recorder,
        )
        acquire_full = make_live_ibex_dae_acquire(
            dae, shared_run=PERIOD_PER_POINT
        )

    stages = []
    for stage_number, (group, accuracy) in enumerate(STEER_ORDER_ACCURACIES):
        group_rows = [row for row in rows if row.type == group]
        indices = [index for index, row in enumerate(rows) if row.type == group]
        metrics = {row.metric for row in group_rows}
        if len(metrics) != 1:
            raise ValueError(f"All {group!r} PVs must use the same metric")
        bounds = [tuple(row.safe_bounds) for row in group_rows]
        metric = _metric_for_group(metrics.pop())
        if len(group_rows) > 1 and (PLOT_LIVE or PLOT_IMAGE_PATH is not None):
            metric = live_metric_monitor(
                bounds=bounds,
                axes=PLOT_AXES,
                alpha=METRIC_ALPHA,
                asymmetry_rebin=PLOT_ASYMMETRY_REBIN,
                asymmetry_time_window_ns=PLOT_ASYMMETRY_TIME_WINDOW_NS,
                min_bin_counts=MIN_BIN_COUNTS,
                grid_size=PLOT_GRID_SIZE,
                minimum_points=PLOT_MINIMUM_POINTS,
                random_state=RANDOM_SEED + stage_number,
                show=PLOT_LIVE,
                save_path=PLOT_IMAGE_PATH,
            )(metric)
            metric.set_axis_labels([row.name for row in group_rows])
        stages.append((
            stage_number, group, accuracy, indices, bounds, metric,
            [row.minimum_step_size for row in group_rows],
            [row.precision for row in group_rows],
            max(row.settle_seconds for row in group_rows),
        ))

    # Matplotlib GUI objects must be created before the RunEngine starts.
    # Stage execution may run on a worker thread, so later boundaries only
    # show and hide these prepared windows.
    for _, _, _, _, _, metric, _, _, _ in stages:
        if hasattr(metric, "open"):
            metric.open()
            metric.deactivate()

    connected_devices = [*motors] + ([] if dae is None else [dae])

    def plan():
        yield from ensure_connected(*connected_devices, timeout=90)
        if dae is not None and PERIOD_PER_POINT:
            yield from bps.stage(dae)
        full_position = np.empty(len(motors), dtype=float)
        for index, motor in enumerate(motors):
            full_position[index] = float((yield from bps.rd(motor)))
        total_measurements = 0
        try:
            for (
                stage_number, group, accuracy, indices, bounds, metric,
                minimum_steps, precisions, settle_seconds,
            ) in stages:
                initial_points = gaussian_initial_coordinates(
                    bounds,
                    full_position[indices].tolist(),
                    n_points=2 * len(indices) + 1,
                    edge_margin_fraction=INITIAL_EDGE_MARGIN_FRACTION,
                    precisions=precisions,
                    random_state=RANDOM_SEED + stage_number,
                )
                required_initial = 2 * len(indices) + 2
                remaining = (
                    MAX_PERIODS - total_measurements
                    if dae is not None and PERIOD_PER_POINT
                    else MAX_MEASUREMENTS
                )
                if remaining < required_initial:
                    raise ValueError(
                        f"Not enough acquisition capacity for {group!r}: "
                        f"need {required_initial}, have {remaining}"
                    )

                def acquire_group(coordinate, *, _indices=indices):
                    full_position[_indices] = coordinate
                    return (yield from acquire_full(full_position.copy()))

                print(f"==============================Stage {stage_number + 1}/{len(stages)}==============================")
                print(f"NOW OPTIMISING {group} with accuracy {accuracy:.3g}")
                print(f"Moving motors {[motors[index].name for index in indices]}")

                if len(indices) == 1:
                    print(
                        "Live plots aren't possible for 1D optimisations; "
                        "continuing without a live plot for this stage."
                    )
                elif hasattr(metric, "activate"):
                    metric.activate()
                try:
                    result = yield from minimize_plan(
                        motors=[motors[index] for index in indices],
                        bounds=bounds,
                        acquire=acquire_group,
                        extract_metrics=metric,
                        accuracy=accuracy,
                        initial_coordinates=initial_points,
                        initial_repeats=2,
                        max_measurements=min(MAX_MEASUREMENTS, remaining),
                        max_acquisitions=(
                            remaining if dae is not None and PERIOD_PER_POINT else None
                        ),
                        minimum_step_sizes=minimum_steps,
                        precisions=precisions,
                        settle_seconds=settle_seconds,
                        final_position="best_measured",
                        random_state=RANDOM_SEED + stage_number,
                        verbose=VERBOSE,
                        minimum_estimate_points=MINIMUM_ESTIMATE_POINTS,
                        benchmark_recorder=recorder,
                    )
                finally:
                    if hasattr(metric, "deactivate"):
                        metric.deactivate()
                total_measurements += result.n_measurements
                for index, motor in enumerate(motors):
                    full_position[index] = float((yield from bps.rd(motor)))
        finally:
            if dae is not None and PERIOD_PER_POINT:
                yield from bps.unstage(dae)
        print("Final PV settings:")
        for row, value in zip(rows, full_position):
            print(f"  {row.name} ({row.setpoint_pv or 'synthetic'}): {value:.6g}")
        print("Total measurements:", total_measurements)
        return full_position

    return plan()
