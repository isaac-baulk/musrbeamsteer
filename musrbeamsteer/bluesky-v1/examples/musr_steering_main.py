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
from time import perf_counter

import numpy as np

from blueskysteercore.acquisition import make_live_ibex_dae_acquire, make_synthetic_acquire
from blueskysteercore.benchmarking import BenchmarkRecorder, duration_as_dict
from blueskysteercore.bluesky_integration import minimize_plan
from blueskysteercore.gp_kg_minimizer import gaussian_initial_coordinates
from blueskysteercore.isis_dae import AcquisitionDuration, make_live_dae
from blueskysteercore.musr_metrics import (
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
benchmark_acq = False  # Enable DAE timing and configuration logging.

INSTRUMENT = "MUSR_LONGITUDINAL" #other options: MUSR_TRANSVERSE, EMU, HIFI

# MUSR steering magnet EPICS PVs (Channel Access by default).
STEER_1_SETPOINT_PV = "IN:MUSR:KEPCO_04:CURRENT:SP"
STEER_1_READBACK_PV = "IN:MUSR:KEPCO_04:CURRENT"
STEER_2_SETPOINT_PV = "IN:MUSR:KEPCO_05:CURRENT:SP"
STEER_2_READBACK_PV = "IN:MUSR:KEPCO_05:CURRENT"
#QUADRAPOLAR_PVs are from MUONFE
# UQ13B_VOLT, UQ13B_POWER, UQ13B_CURR and same for 14B theres also M_SILTS
FOCUS_1_SETPOINT_PV = "IN:MUONFE:GENESYS_03:1:CURR:SP"
FOCUS_1_READBACK_PV = "IN:MUONFE:GENESYS_03:1:CURR"
FOCUS_2_SETPOINT_PV = "IN:MUONFE:GENESYS_03:2:CURR:SP"
FOCUS_2_READBACK_PV = "IN:MUONFE:GENESYS_03:2:CURR"


# Approved test bounds in the magnets' readback units (normally amperes).
# Review these values locally before enabling real magnet movement.
SAFE_BOUNDS = ((-0.5, 0.5), (-1.5, 1.5))
READBACK_TOLERANCES = (0.01, 0.01)
MOVE_TIMEOUT_SECONDS = 30.0
SETTLE_SECONDS = 1.0
MINIMUM_STEP_SIZES = None  # optionally set one minimum increment per magnet
CURRENT_PRECISIONS = [3, 3]  # decimal places per magnet

SYNTHETIC_MOTORS = (
    SyntheticMotor(
        name="MUSR_STEER_1",
        setpoint_pv=STEER_1_SETPOINT_PV,
        readback_pv=STEER_1_READBACK_PV,
        safe_bounds=SAFE_BOUNDS[0],
        readback_tolerance=READBACK_TOLERANCES[0],
        timeout=MOVE_TIMEOUT_SECONDS,
    ),
    SyntheticMotor(
        name="MUSR_STEER_2",
        setpoint_pv=STEER_2_SETPOINT_PV,
        readback_pv=STEER_2_READBACK_PV,
        safe_bounds=SAFE_BOUNDS[1],
        readback_tolerance=READBACK_TOLERANCES[1],
        timeout=MOVE_TIMEOUT_SECONDS,
    ),
)

# Ask for INITIAL_GUESS at startup. Set e.g. [0.0, -0.2] to skip the prompt.
INITIAL_GUESS = [0, 0]  # prompt: [p1, ..., pN]
INITIAL_DESIGN_POINTS = None  # default: 1 + 2*N Gaussian points
INITIAL_EDGE_MARGIN_FRACTION = 0.10  # keep initial points 10% from each edge
# Each dimension is sampled with sigma=distance to the nearest bound / 3.
# Initial draws are rejected, not clipped, when outside the interior margin.

# Use the SAME duration for synthetic and real modes.
# DURATION = AcquisitionDuration.good_frames(500)
DURATION = AcquisitionDuration.events(1_000_000)
# DURATION = AcquisitionDuration.seconds(10.0)

# N-dimensional simulated sample centre / Gaussian spatial curvature.
# These describe the synthetic beam, NOT the user's initial guess.
CENTER = [0.1, -0.3]
CURVATURE = [3.0, 3.0]  # inverse-square of current units

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
METRIC_NAME = "integrated_asymmetry"  # or "pca" or "asym_squared_count_rate"
METRIC_TIME_WINDOW_NS = (0.0, 30000.0)  # synthetic t0=0; calibrate for live DAE
METRIC_REBIN = 1  # native 16 ns bins; set 8 for ~128 ns channels
METRIC_ALPHA = 1.0  # calibrate for real MUSR measurements
MIN_BIN_COUNTS = 10.0

MAX_MEASUREMENTS = 50
INITIAL_REPEATS = 2
# 0: quiet, 1: measurements + best measured, 2: also show GP estimate/diagnostics.
VERBOSE = 2
MINIMUM_ESTIMATE_POINTS = 1024  # used only when VERBOSE=2 (no motor moves)
# ================================================================



def _parse_initial_guess(value: str, n_dimensions: int) -> list[float]:
    """Parse '[p1, ..., pN]', 'p1,p2' or 'p1 p2' from Windows CMD."""
    raw = value.strip()
    if raw.startswith("[") or raw.startswith("("):
        guess = ast.literal_eval(raw)
    else:
        guess = [part for part in re.split(r"[,\s]+", raw) if part]
    if not isinstance(guess, (tuple, list)) or len(guess) != n_dimensions:
        raise ValueError(f"Enter exactly {n_dimensions} coordinates")
    try:
        return [float(v) for v in guess]
    except (ValueError, TypeError) as exc:
        raise ValueError("Every initial-guess coordinate must be numeric") from exc


def resolve_initial_guess(*, input_fn=None) -> list[float]:
    """Ask once per execution, validate before connecting to any hardware.

    Set INITIAL_GUESS to a list to disable prompting in unattended runs.
    A rejected guess is re-requested; Ctrl+C / EOF aborts without moving.
    """
    prompt = input if input_fn is None else input_fn
    n_dimensions = len(SAFE_BOUNDS)
    while True:
        if INITIAL_GUESS is None:
            message = (
                f"Initial guess for {n_dimensions} steering currents "
                f"within {SAFE_BOUNDS}, e.g. [0.0, -0.2]: "
            )
            raw = prompt(message)
            try:
                guess = _parse_initial_guess(raw, n_dimensions)
            except (ValueError, SyntaxError) as exc:
                print(f"Invalid initial guess: {exc}")
                continue
        else:
            guess = list(INITIAL_GUESS)

        try:
            # Validate even when no prompt is used. Does not access EPICS.
            gaussian_initial_coordinates(
                SAFE_BOUNDS, guess, n_points=1,
                edge_margin_fraction=INITIAL_EDGE_MARGIN_FRACTION,
                precisions=CURRENT_PRECISIONS, random_state=RANDOM_SEED,
            )
        except (TypeError, ValueError) as exc:
            if INITIAL_GUESS is not None:
                raise ValueError(f"Configured INITIAL_GUESS is invalid: {exc}") from exc
            print(f"Initial guess rejected: {exc}")
            continue
        return guess


def make_simulator():
    """Create a simulator using the current top-level constants.

    functools.partial freezes these parameters per optimisation run; one RNG
    instance generates fresh Poisson noise at repeats. Editing the constants
    above is all that is required to reconfigure the synthetic experiment.
    """
    return partial(
        generate_synthetic_musr_data,
        center=CENTER,
        curvature=CURVATURE,
        bounds=SAFE_BOUNDS,
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
    """Return a Bluesky plan; no EPICS writes until the RunEngine runs it."""
    pvs = (
        ("musr_steering_1", STEER_1_SETPOINT_PV, STEER_1_READBACK_PV),
        ("musr_steering_2", STEER_2_SETPOINT_PV, STEER_2_READBACK_PV),
    )
    if any("REPLACE_WITH" in pv for _, sp, rb in pvs for pv in (sp, rb)):
        raise ValueError("Set the actual steering magnet setpoint/readback PVs")
    if any(lo is None or hi is None for lo, hi in SAFE_BOUNDS):
        raise ValueError("Set SAFE_BOUNDS for both magnets before enabling any movement")
    if any(tol is None for tol in READBACK_TOLERANCES):
        raise ValueError("Set READBACK_TOLERANCES for both magnets")
    if len(SAFE_BOUNDS) != len(CENTER) or len(CENTER) != len(CURVATURE):
        raise ValueError("SAFE_BOUNDS, CENTER and CURVATURE must have the same dimension")
    if any(not lo < c < hi for (lo, hi), c in zip(SAFE_BOUNDS, CENTER)):
        raise ValueError("Set CENTER strictly inside both safe current bounds")
    if not np.isclose(N_TIME_BINS * BIN_WIDTH_NS, MAX_TIME_US * 1000):
        raise ValueError("Time range must be an integer number of histogram bins")

    # Prompt and generate all initial samples BEFORE any EPICS connection/write.
    initial_guess = resolve_initial_guess()
    initial_points = gaussian_initial_coordinates(
        SAFE_BOUNDS, initial_guess,
        n_points=INITIAL_DESIGN_POINTS,
        edge_margin_fraction=INITIAL_EDGE_MARGIN_FRACTION,
        precisions=CURRENT_PRECISIONS,
        random_state=RANDOM_SEED,
    )
    if VERBOSE:
        nearest = np.minimum(
            np.asarray(initial_guess) - np.asarray(SAFE_BOUNDS)[:, 0],
            np.asarray(SAFE_BOUNDS)[:, 1] - np.asarray(initial_guess),
        )
        print(f"Initial guess: {initial_guess}")
        print(f"Initial Gaussian sigma (nearest edge / 3): {(nearest / 3).tolist()}")
        print(f"Initial interior samples ({len(initial_points)}):")
        for index, x in enumerate(initial_points, start=1):
            print(f"  {index:2d}: {np.array2string(x, precision=6)}")

    from ophyd_async.plan_stubs import ensure_connected
    import bluesky.plan_stubs as bps

    if USE_SYNTHETIC_MOTORS:
        motors = [motor.build() for motor in SYNTHETIC_MOTORS]
    else:
        motors = [
            make_setpoint_readback_axis(
                name=name, setpoint_pv=sp, readback_pv=rb,
                tolerance=tol, timeout=MOVE_TIMEOUT_SECONDS, limits=bounds,
            )
            for (name, sp, rb), tol, bounds in zip(pvs, READBACK_TOLERANCES, SAFE_BOUNDS)
        ]

    recorder = BenchmarkRecorder() if benchmark_acq else None
    if recorder is not None:
        recorder.log(
            "configuration",
            period_per_point=PERIOD_PER_POINT,
            steering_pvs={
                "steer_1_setpoint": STEER_1_SETPOINT_PV,
                "steer_1_readback": STEER_1_READBACK_PV,
                "steer_2_setpoint": STEER_2_SETPOINT_PV,
                "steer_2_readback": STEER_2_READBACK_PV,
            },
            instrument=INSTRUMENT,
            use_synthetic_data=USE_SYNTHETIC_DATA,
            use_synthetic_motors=USE_SYNTHETIC_MOTORS,
            synthetic={
                "center": CENTER,
                "curvature": CURVATURE,
                "kt_delta_us_inv": KT_DELTA_US_INV,
                "initial_asymmetry": INITIAL_ASYMMETRY,
                "muon_lifetime_us": MUON_LIFETIME_US,
                "counts_per_good_frame": COUNTS_PER_GOOD_FRAME,
                "edge_rate_fraction": EDGE_RATE_FRACTION,
                "good_frames_per_second": GOOD_FRAMES_PER_SECOND,
                "bin_width_ns": BIN_WIDTH_NS,
                "max_time_us": MAX_TIME_US,
                "n_time_bins": N_TIME_BINS,
                "random_seed": RANDOM_SEED,
            } if USE_SYNTHETIC_DATA else None,
            settle_seconds=SETTLE_SECONDS,
            duration=duration_as_dict(DURATION),
            metric={
                "name": METRIC_NAME,
                "time_window_ns": METRIC_TIME_WINDOW_NS,
                "rebin": METRIC_REBIN,
                "alpha": METRIC_ALPHA,
                "min_bin_counts": MIN_BIN_COUNTS,
            },
            max_measurements=MAX_MEASUREMENTS,
            initial_repeats=INITIAL_REPEATS,
        )

    if USE_SYNTHETIC_DATA:
        acquire = make_synthetic_acquire(make_simulator())
        connected_devices = motors
    else:
        if PERIOD_PER_POINT:
            if MAX_MEASUREMENTS > MAX_PERIODS:
                raise ValueError(
                    f"MAX_MEASUREMENTS ({MAX_MEASUREMENTS}) exceeds the shared DAE "
                    f"period limit ({MAX_PERIODS})"
                )
        dae = make_live_dae(
            duration=DURATION,
            instrument=INSTRUMENT,
            period_per_point=PERIOD_PER_POINT,
            read_time_axis=True,
            benchmark_recorder=recorder,
        )
        acquire = make_live_ibex_dae_acquire(dae, shared_run=PERIOD_PER_POINT)
        connected_devices = [*motors, dae]

    metric_options = dict(
        alpha=METRIC_ALPHA,
        time_window_ns=METRIC_TIME_WINDOW_NS,
        rebin=METRIC_REBIN,
        min_bin_counts=MIN_BIN_COUNTS,
    )
    if METRIC_NAME == "integrated_asymmetry":
        # Original project's weighted asymmetry metric: smaller at the KT centre.
        metric = integrated_asymmetry_metric(**metric_options)
    elif METRIC_NAME == "pca":
        # Dataset-dependent: recompute ALL PC1 scores when new data arrive.
        metric = pca_asymmetry_metric(**metric_options)
    elif METRIC_NAME == "asym_squared_count_rate":
        # Inverse-variance-weighted integrated asymmetry squared times count rate.
        metric = asym_squared_count_rate_metric(**metric_options)
    else:
        raise ValueError("METRIC_NAME must be 'integrated_asymmetry' or 'pca' or 'asym_squared_count_rate'")

    def plan():
        yield from ensure_connected(*connected_devices, timeout=90)
        if PERIOD_PER_POINT and not USE_SYNTHETIC_DATA:
            yield from bps.stage(dae)
        try:
            return (yield from minimize_plan(
                motors=motors,
                bounds=SAFE_BOUNDS,
                acquire=acquire,
                extract_metrics=metric,
                initial_coordinates=initial_points,
                initial_repeats=INITIAL_REPEATS,
                max_measurements=MAX_MEASUREMENTS,
                max_acquisitions=(
                    MAX_PERIODS
                    if PERIOD_PER_POINT and not USE_SYNTHETIC_DATA
                    else None
                ),
                minimum_step_sizes=MINIMUM_STEP_SIZES,
                precisions=CURRENT_PRECISIONS,
                settle_seconds=SETTLE_SECONDS,
                final_position="best_measured",
                random_state=RANDOM_SEED,
                verbose=VERBOSE,
                minimum_estimate_points=MINIMUM_ESTIMATE_POINTS,
                benchmark_recorder=recorder,
            ))
        finally:
            if PERIOD_PER_POINT and not USE_SYNTHETIC_DATA:
                save_started = perf_counter()
                yield from bps.unstage(dae)
                if recorder is not None:
                    recorder.record("dae_end_save", perf_counter() - save_started)
            if recorder is not None:
                recorder.log_summary(
                    print_run_per_point=(
                        not PERIOD_PER_POINT and not USE_SYNTHETIC_DATA
                    )
                )

    return plan()


# At MUSR, in a session where the Bluesky RunEngine `RE` is already defined:
#
# from examples.musr_steering_synthetic import make_plan
# result = RE(make_plan())
#
# The exact method for accessing a plan's return value depends on the
# instrument RunEngine configuration (some enable RunEngineResult).
# With both synthetic switches true, neither EPICS nor the DAE is contacted.
