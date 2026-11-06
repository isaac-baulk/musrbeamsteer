"""Bluesky N-dimensional experimental minimisation.

The optimiser and the acquisition/reduction systems are deliberately separate:
* ``minimize_plan`` owns RunEngine-compatible device moves and optimisation.
* ``acquire`` is a user-provided Bluesky plan returning a measurement record.
* ``extract_metrics`` recalculates one scalar per *all* measurements; this is
  important for PCA where historical scores change as more runs are added.

No Genie, Mantid, Ophyd or IBEX imports are required to import this module.
Only running a plan requires Bluesky; Mantid/IBEX integrations are optional.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from time import monotonic
from typing import Any

import numpy as np

from .gp_kg_minimizer import (
    Observation, SearchResult, _validate_bounds, _validate_minimum_step_sizes,
    _validate_precisions, _validate_coordinate, _round_coordinate,
    default_initial_coordinates,
    _recompute_all_metrics, estimate_noise_variance,
    next_coordinate_with_warning_recovery, _model_minimum_or_best_measurement,
    random_exploration_after_gp_warning, RandomExplorationExhausted,
    _print_minimum_estimate, _print_best_measured, _validate_verbose,
)


def minimize_plan(
    motors: Sequence[Any],
    bounds: Sequence[Sequence[float]],
    acquire: Callable[[np.ndarray], Any],
    extract_metrics: Callable[..., Sequence[float]],
    *,
    accuracy: float = 0.0001,
    initial_coordinates: Iterable[Sequence[float]] | None = None,
    initial_repeats: int = 2,
    max_measurements: int = 30,
    max_acquisitions: int | None = None,
    minimum_step_sizes: Sequence[float] | None = None,
    precisions: Sequence[int] | None = None,
    settle_seconds: float = 0.0,
    random_state: int = 12345,
    n_candidate_points: int = 192,
    n_decision_points: int = 1024,
    progress_callback: Callable[[Sequence[Observation], float | None], None] | None = None,
    benchmark_recorder: Any | None = None,
    restore_on_error: bool = True,
    final_position: str = "best_measured",
    verbose: int | bool = 2,
    minimum_estimate_points: int = 1024,
):
    """Optimise arbitrary Ophyd positioners using RunEngine-compatible plans.

    ``acquire(coordinate)`` must return a Bluesky generator plan whose return
    value is an opaque measurement record (e.g. an external NeXus run number).
    ``extract_metrics(measurements, first_set=False)`` recomputes all scores.

    ``final_position``: 'best_measured', 'model', or 'last'. The model minimum
    is *predicted*, not necessarily measured; 'best_measured' is safer.
    A GP fit warning (including constant_value/length_scale bound warnings)
    makes the next acquisition a random, unmeasured point in ``bounds``.
    The plan refits after each random measurement and resumes KG selection
    once fitting is warning-free. ``max_measurements`` is a hard limit.
    ``max_acquisitions`` optionally rejects a measurement budget larger than
    the capacity of the acquisition backend before any device is moved.
    If the final GP still warns, the best measured point is returned.
    On exceptions the starting motor values are restored when possible.
    ``verbose`` controls progress reporting: 0 = quiet, 1 = measurements,
    best measured coordinate and fit warnings, 2 = level 1 plus GP-predicted
    minimum and proposal diagnostics. True/False map to 2/0 for compatibility.
    Only level 2 runs the extra progress GP fits using
    ``minimum_estimate_points`` Sobol points; these never move devices. A GP
    warning suppresses untrusted model estimates. No nested RunEngine or
    ``genie_python`` commands are used.
    """
    import bluesky.plan_stubs as bps

    b = _validate_bounds(bounds)
    verbose = _validate_verbose(verbose)
    if len(motors) != len(b) or not motors:
        raise ValueError("motors and bounds must have matching nonzero lengths")
    if len({id(m) for m in motors}) != len(motors):
        raise ValueError("motors must be distinct")
    if accuracy <= 0 or initial_repeats < 2 or max_measurements < 2:
        raise ValueError("accuracy > 0, initial_repeats >= 2 and max_measurements >= 2 required")
    if max_acquisitions is not None:
        if (
            isinstance(max_acquisitions, bool)
            or not isinstance(max_acquisitions, int)
            or max_acquisitions < 1
        ):
            raise ValueError("max_acquisitions must be a positive integer or None")
        if max_measurements > max_acquisitions:
            raise ValueError(
                f"max_measurements ({max_measurements}) exceeds the acquisition limit "
                f"({max_acquisitions})"
            )
    if settle_seconds < 0:
        raise ValueError("settle_seconds must be >= 0")
    if final_position not in ("best_measured", "model", "last"):
        raise ValueError("final_position must be 'best_measured', 'model', or 'last'")
    if (isinstance(minimum_estimate_points, bool)
            or not isinstance(minimum_estimate_points, int)
            or not 2 <= minimum_estimate_points <= 65536):
        raise ValueError("minimum_estimate_points must be an integer from 2 to 65536")

    steps = _validate_minimum_step_sizes(minimum_step_sizes, len(motors))
    digits = _validate_precisions(precisions, len(motors))
    starts = initial_coordinates if initial_coordinates is not None else default_initial_coordinates(b)
    starts = [_round_coordinate(_validate_coordinate(x, b), digits, b) for x in starts]
    if not starts:
        raise ValueError("at least one initial coordinate required")

    observations: list[Observation] = []
    converged = False
    kg = float("nan")
    random_explorations = 0
    gp_warnings: list[str] = []
    original = []
    for motor in motors:
        original.append(float((yield from bps.rd(motor))))
    # If the start lies outside the approved search space, abort before moving.
    # Restoring to that location later would otherwise violate our own limits.
    _validate_coordinate(original, b)

    def move_to(coordinate):
        safe = _validate_coordinate(coordinate, b)
        args = [item for motor, x in zip(motors, safe) for item in (motor, float(x))]
        move_started = monotonic()
        yield from bps.mv(*args)
        if benchmark_recorder is not None:
            benchmark_recorder.record("epics_move", monotonic() - move_started)
        if settle_seconds:
            yield from bps.sleep(settle_seconds)

    def acquire_one(coordinate):
        yield from move_to(coordinate)
        measurement = yield from acquire(np.asarray(coordinate, dtype=float).copy())
        if measurement is None:
            raise ValueError("acquisition plan must return a measurement record")
        observations.append(Observation(coordinate=np.asarray(coordinate, dtype=float).copy(), measurement=measurement))

    def recalculate(first_set=False):
        _recompute_all_metrics(observations, extract_metrics, first_set)
        noise = None
        try:
            noise = math.sqrt(estimate_noise_variance(observations))
        except RuntimeError:
            pass
        if progress_callback is not None:
            progress_callback(observations, noise)
        if verbose:
            current = observations[-1]
            print(f"{len(observations):3d}: x={current.coordinate}, metric={current.metric:.8g}, noise={noise}")
            if noise is not None:
                if verbose >= 2:
                    _print_minimum_estimate(
                        observations, b,
                        random_state=random_state + len(observations),
                        n_points=minimum_estimate_points,
                    )
                else:
                    _print_best_measured(observations)

    try:
        n_initial = min(max_measurements, len(starts) + initial_repeats - 1)
        for i, x in enumerate(starts):
            for _ in range(initial_repeats if i == 0 else 1):
                if len(observations) >= max_measurements:
                    break
                yield from acquire_one(x)
                if verbose and len(observations) < n_initial:
                    print(
                        f"{len(observations):3d}: x={np.array2string(x, precision=6)}, "
                        "estimated minimum pending (collecting initial design)"
                    )
            if len(observations) >= max_measurements:
                break

        recalculate(first_set=True)
        estimate_noise_variance(observations)

        while len(observations) < max_measurements:
            try:
                proposal = next_coordinate_with_warning_recovery(
                    observations, b, steps, digits,
                    n_candidate_points=n_candidate_points,
                    n_decision_points=n_decision_points,
                    random_state=random_state + len(observations),
                )
            except RandomExplorationExhausted as exc:
                # A fully sampled discretised search space cannot supply
                # another novel point, even if the GP keeps warning.
                if verbose:
                    print(f"Random exploration stopped: {exc}")
                break

            # Check the final model before declaring convergence: a different
            # GP fit seed can reveal warnings even when the KG fit was clean.
            if not proposal.random_exploration and proposal.kg < accuracy:
                _, _, fit_ok, fit_warnings = _model_minimum_or_best_measurement(
                    observations, b, random_state=random_state + 99999,
                )
                if fit_ok:
                    kg = proposal.kg
                    converged = True
                    break
                try:
                    proposal = random_exploration_after_gp_warning(
                        observations, b, steps, digits, fit_warnings,
                        random_state=random_state + len(observations),
                    )
                except RandomExplorationExhausted:
                    if verbose:
                        print("No new random coordinate after final-model fit warning")
                    break

            coordinate, kg = proposal.coordinate, proposal.kg
            if proposal.random_exploration:
                random_explorations += 1
                gp_warnings.extend(proposal.warning_messages)
                if verbose:
                    for message in proposal.warning_messages:
                        print(f"GP fit warning -> random exploration: {message}")
            if verbose >= 2:
                print(
                    "     next="
                    f"{np.array2string(coordinate, precision=6)}, "
                    f"strategy={'random' if proposal.random_exploration else 'KG'}, "
                    f"KG={kg:.6g}, noise_sigma={proposal.noise_std:.6g}"
                )
            yield from acquire_one(coordinate)
            recalculate()

        _recompute_all_metrics(observations, extract_metrics)
        model_position, model_value, model_fit_ok, final_warnings = _model_minimum_or_best_measurement(
            observations, b, random_state=random_state + 99999
        )
        gp_warnings.extend(final_warnings)
        if final_warnings and verbose:
            print("Final GP fit still warns: using the best measured point")
        model_position = _round_coordinate(model_position, digits, b)
        noise = math.sqrt(estimate_noise_variance(observations))
        result = SearchResult(
            coordinate=model_position, value=model_value,
            estimated_noise_std=noise, n_measurements=len(observations),
            converged=converged and model_fit_ok, observations=observations, kg=kg,
            random_explorations=random_explorations,
            model_fit_ok=model_fit_ok, gp_warnings=tuple(gp_warnings),
        )
        if final_position == "model":
            yield from move_to(model_position)
        elif final_position == "best_measured":
            best = min(observations, key=lambda o: o.metric)
            yield from move_to(best.coordinate)
        return result
    except Exception:
        if restore_on_error:
            try:
                yield from move_to(original)
            except Exception:
                # Preserve original failure. Operators should verify hardware state.
                print("WARNING: Could not restore all motors to their starting positions")
        raise


# Backwards compatibility with the original example.
beam_steering_plan = minimize_plan
from .acquisition import make_ibex_dae_acquire, make_bluesky_acquire
