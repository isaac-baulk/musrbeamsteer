#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 08:59:59 2026

@author: hwh96151
"""


"""
gpkg.py
=======

Generic Gaussian-Process + approximate Knowledge-Gradient minimizer for a
small (typically 1-4 dimensional) noisy parameter space.

This module contains NO instrument-specific code and NO genie_python import.

The experiment is supplied through two callbacks:

    measurement = get_measurement(coordinate)
    metrics = extract_metrics(all_measurements)

extract_metrics() receives EVERY measurement collected so far and must
return one scalar metric per measurement, in the same order. This supports
global analyses such as PCA where adding a new point can change old metrics.

The metric must be scalar and LOWER must mean BETTER.

Unknown noise
-------------
The absolute noise level does not need to be known.  The code estimates a
single observation-noise variance from repeated measurements at identical
coordinates.  This is appropriate when every measurement uses the same
statistical exposure (for example exactly 1 million detector events).

At least one coordinate must therefore be repeated.  The normal choice is
initial_repeats=2.

Dependencies
------------
    numpy
    scipy
    scikit-learn
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import solve_triangular
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


@dataclass
class Observation:
    """One measured coordinate, its raw measurement, and derived metric."""

    coordinate: np.ndarray
    measurement: Any = None
    metric: float = np.nan


@dataclass
class SearchResult:
    """Final result and complete observation history of a search."""

    coordinate: np.ndarray
    value: float
    estimated_noise_std: float
    n_measurements: int
    converged: bool
    observations: list[Observation]
    kg: float
    random_explorations: int = 0
    model_fit_ok: bool = True
    gp_warnings: tuple[str, ...] = ()


class GPFitWarning(RuntimeError):
    """GP fitting emitted warnings and its predictions should not be trusted.

    Unlike a fit exception, this means scikit-learn produced a fitted model
    but reported a warning (for example a ConstantKernel or Matern parameter
    reaching its bound). The caller should acquire additional data.
    """

    def __init__(self, messages: Sequence[str]):
        self.messages = tuple(messages)
        super().__init__("; ".join(self.messages))


class RandomExplorationExhausted(RuntimeError):
    """No novel random point could satisfy the experiment's constraints."""


@dataclass(frozen=True)
class NextCoordinate:
    """A KG recommendation or a warning-triggered random exploration."""

    coordinate: np.ndarray
    kg: float
    noise_std: float
    random_exploration: bool = False
    warning_messages: tuple[str, ...] = ()


def _validate_bounds(bounds: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate bounds and return them as a floating-point array.

    Parameters
    ----------
    bounds : sequence of sequence of float
        One finite ``(lower, upper)`` pair per dimension.
    Returns
    -------
    numpy.ndarray
        A two-dimensional floating-point bounds array.
    Raises
    ------
    ValueError
        If the bounds shape or values are invalid.
    """
    b = np.asarray(bounds, dtype=float)

    if b.ndim != 2 or b.shape[1] != 2 or b.shape[0] < 1:
        raise ValueError(
            "bounds must be [(min_1, max_1), ..., (min_N, max_N)]."
        )

    if not np.all(np.isfinite(b)):
        raise ValueError("bounds contain a non-finite value.")

    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError(
            "For every parameter, the upper bound must be greater than "
            "the lower bound."
        )

    return b


def _validate_coordinate(
    coordinate: Sequence[float],
    bounds: np.ndarray,
) -> np.ndarray:
    """Validate one coordinate against the supplied bounds.

    Parameters
    ----------
    coordinate : sequence of float
        Coordinate values in physical units.
    bounds : numpy.ndarray
        Validated lower and upper bounds.
    Returns
    -------
    numpy.ndarray
        The coordinate as a floating-point NumPy array.
    Raises
    ------
    ValueError
        If its dimension, finiteness, or range is invalid.
    """
    x = np.asarray(coordinate, dtype=float)

    if x.shape != (bounds.shape[0],):
        raise ValueError(
            f"Expected a {bounds.shape[0]}-D coordinate, got shape {x.shape}."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("coordinate contains a non-finite value.")

    if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
        raise ValueError(
            f"Coordinate {x.tolist()} is outside bounds {bounds.tolist()}."
        )

    return x


def _validate_minimum_step_sizes(
    minimum_step_sizes: Optional[Sequence[float]],
    dimension: int,
) -> np.ndarray:
    """Validate minimum movement sizes for each coordinate dimension.

    Parameters
    ----------
    minimum_step_sizes : sequence of float or None
        Optional non-negative step sizes.
    dimension : int
        Number of coordinate dimensions.
    Returns
    -------
    numpy.ndarray
        A floating-point array of step sizes.
    Raises
    ------
    ValueError
        If a supplied step-size sequence is invalid.
    """
    if minimum_step_sizes is None:
        return np.zeros(dimension, dtype=float)

    steps = np.asarray(minimum_step_sizes, dtype=float)
    if steps.shape != (dimension,):
        raise ValueError(
            f"Expected {dimension} minimum step sizes, got shape {steps.shape}."
        )
    if not np.all(np.isfinite(steps)) or np.any(steps < 0):
        raise ValueError(
            "minimum_step_sizes must contain only finite, non-negative values."
        )
    return steps


def _validate_precisions(
    precisions: Optional[Sequence[int]],
    dimension: int,
) -> np.ndarray:
    """Validate decimal precision settings for each dimension.

    Parameters
    ----------
    precisions : sequence of int or None
        Optional non-negative decimal-place counts. ``None`` means no rounding.
    dimension : int
        Number of coordinate dimensions.
    Returns
    -------
    numpy.ndarray
        An integer array of precisions.
    Raises
    ------
    ValueError
        If a supplied precision sequence is invalid.
    """
    if precisions is None:
        # -1 is an internal sentinel for unrounded floating-point coordinates.
        # A default of zero would round -1.5 to -2, exceeding hardware limits.
        return np.full(dimension, -1, dtype=int)

    values = np.asarray(precisions)
    if values.shape != (dimension,) or not np.all(
        np.equal(values, np.floor(values))
    ):
        raise ValueError(
            f"Expected {dimension} non-negative integer precisions."
        )
    values = values.astype(int)
    if np.any(values < 0):
        raise ValueError("precisions must be non-negative integers.")
    return values


def _round_coordinate(
    coordinate: Sequence[float],
    precisions: np.ndarray,
    bounds: np.ndarray | None = None,
) -> np.ndarray:
    """Round requested coordinates without leaving configured bounds.

    Precision -1 is internal shorthand for no rounding (the public ``None``).
    Integer precision 0 explicitly requests rounding to whole units.
    Clipping *after* rounding matters at half-integer limits: -1.5 rounds
    to -2 with ``round(value, 0)``, which could move hardware out of range.
    """
    rounded = np.asarray(
        [float(value) if precision < 0 else round(float(value), int(precision))
         for value, precision in zip(coordinate, precisions)],
        dtype=float,
    )
    if bounds is not None:
        rounded = np.clip(rounded, bounds[:, 0], bounds[:, 1])
    return rounded


def _apply_minimum_step_sizes(
    coordinate: Sequence[float],
    previous_coordinate: Sequence[float],
    minimum_step_sizes: np.ndarray,
    precisions: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Apply minimum movement and precision constraints to a coordinate.

    Parameters
    ----------
    coordinate : sequence of float
        Requested next coordinate.
    previous_coordinate : sequence of float
        Coordinate of the preceding observation.
    minimum_step_sizes : numpy.ndarray
        Minimum movement per dimension.
    precisions : numpy.ndarray
        Decimal-place count per dimension.
    bounds : numpy.ndarray
        Validated coordinate bounds.
    Returns
    -------
    numpy.ndarray
        A rounded, clipped coordinate satisfying the movement constraints.
    Raises
    ------
    ValueError
        If either coordinate is outside the supplied bounds.
    """
    requested = _validate_coordinate(coordinate, bounds)
    adjusted = _round_coordinate(requested, precisions)
    previous = _round_coordinate(
        _validate_coordinate(previous_coordinate, bounds),
        precisions,
    )
    delta = requested - previous
    small_steps = (minimum_step_sizes > 0) & (delta != 0) & (
        np.abs(delta) < minimum_step_sizes
    )
    # Do not invent a precision grid when the caller requested no rounding.
    required_steps = minimum_step_sizes.copy()
    quantised = precisions >= 0
    if np.any(quantised):
        grid_step = np.power(10.0, -precisions[quantised].astype(float))
        required_steps[quantised] = (
            np.ceil(minimum_step_sizes[quantised] / grid_step - 1e-12)
            * grid_step
        )
    adjusted[small_steps] = (
        previous[small_steps]
        + np.sign(delta[small_steps]) * required_steps[small_steps]
    )
    return np.clip(
        _round_coordinate(adjusted, precisions),
        bounds[:, 0],
        bounds[:, 1],
    )


def _to_unit(x: Sequence[float], bounds: np.ndarray) -> np.ndarray:
    """Map a physical coordinate into the unit hypercube.

    Parameters
    ----------
    x : sequence of float
        Physical coordinate.
    bounds : numpy.ndarray
        Validated physical bounds.
    Returns
    -------
    numpy.ndarray
        Coordinate values scaled to ``[0, 1]``.
    """
    x = _validate_coordinate(x, bounds)
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def _from_unit(u: Sequence[float], bounds: np.ndarray) -> np.ndarray:
    """Map unit-hypercube values back into physical coordinates.

    Parameters
    ----------
    u : sequence of float
        Unit-hypercube coordinate values.
    bounds : numpy.ndarray
        Physical lower and upper bounds.
    Returns
    -------
    numpy.ndarray
        A physical coordinate clipped to the unit hypercube's bounds.
    """
    u = np.asarray(u, dtype=float)
    return bounds[:, 0] + np.clip(u, 0.0, 1.0) * (
        bounds[:, 1] - bounds[:, 0]
    )


def default_initial_coordinates(
    bounds: Sequence[Sequence[float]],
) -> list[np.ndarray]:
    """
    Return a simple dimension-independent initial design.

    It contains:
      * the centre of the parameter space;
      * one point at the lower boundary of each axis;
      * one point at the upper boundary of each axis;

    while all other coordinates remain at their centre.

    Number of points = 1 + 2*N.

    Examples:
        N=1 -> 3 points
        N=2 -> 5 points
        N=4 -> 9 points

    With initial_repeats=2 this uses 6, 10 and 18 measurements respectively.

    Parameters
    ----------
    bounds : sequence of sequence of float
        One finite ``(lower, upper)`` pair per dimension.
    Returns
    -------
    list of numpy.ndarray
        The centre point plus one lower and upper axial point per dimension.
    Raises
    ------
    ValueError
        If ``bounds`` is invalid.
    """
    b = _validate_bounds(bounds)
    centre = np.mean(b, axis=1)

    points = [centre.copy()]

    for dimension in range(b.shape[0]):
        low = centre.copy()
        high = centre.copy()

        low[dimension] = b[dimension, 0]
        high[dimension] = b[dimension, 1]

        points.extend([low, high])

    return points



def gaussian_initial_coordinates(
    bounds: Sequence[Sequence[float]],
    initial_guess: Sequence[float],
    *,
    n_points: int | None = None,
    edge_margin_fraction: float = 0.02,
    precisions: Sequence[int] | None = None,
    random_state: int = 12345,
    max_attempts_per_point: int = 4096,
) -> list[np.ndarray]:
    """Draw a reproducible interior initial design around an operator guess.

    Each parameter is sampled independently from a normal distribution with

        sigma_i = min(guess_i - lower_i, upper_i - guess_i) / 3.

    Thus the *nearest* boundary is 3 standard deviations from the guessed
    mean along each axis. Samples outside the interior region are REJECTED
    rather than clipped to an edge (clipping would accumulate edge points).
    The configurable fraction of each full axis range reserved at either end
    applies only to the *initial design*. It is not a hardware interlock and
    does not constrain later KG or GP-warning random exploration.

    By default, draw 1 + 2 * N distinct N-dimensional points. All points,
    including the first (which the optimiser repeats to estimate noise), are
    Gaussian samples. Precision quantisation must not move a point into the
    excluded margin, and repeated rounded points are rejected.
    """
    b = _validate_bounds(bounds)
    guess = _validate_coordinate(initial_guess, b)
    n_dimensions = len(guess)
    if n_points is None:
        n_points = 1 + 2 * n_dimensions
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    if not np.isfinite(edge_margin_fraction) or not 0 <= edge_margin_fraction < 0.5:
        raise ValueError("edge_margin_fraction must be finite and in [0, 0.5)")
    if isinstance(max_attempts_per_point, bool) or not isinstance(max_attempts_per_point, int) or max_attempts_per_point < 1:
        raise ValueError("max_attempts_per_point must be a positive integer")
    digits = _validate_precisions(precisions, n_dimensions)
    span = b[:, 1] - b[:, 0]
    lower_interior = b[:, 0] + edge_margin_fraction * span
    upper_interior = b[:, 1] - edge_margin_fraction * span
    if np.any(guess <= lower_interior) or np.any(guess >= upper_interior):
        raise ValueError(
            "initial_guess must lie strictly inside the initial-design "
            f"{100 * edge_margin_fraction:g}% edge margin: "
            f"{list(zip(lower_interior, upper_interior))}"
        )
    sigma = np.minimum(guess - b[:, 0], b[:, 1] - guess) / 3.0
    rng = np.random.default_rng(random_state)
    points: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for _ in range(int(n_points)):
        for attempt in range(max_attempts_per_point):
            proposed = rng.normal(loc=guess, scale=sigma)
            if np.any(proposed <= lower_interior) or np.any(proposed >= upper_interior):
                continue
            rounded = _round_coordinate(proposed, digits, b)
            if np.any(rounded <= lower_interior) or np.any(rounded >= upper_interior):
                continue
            key = tuple(float(v) for v in rounded)
            if key in seen:
                continue
            seen.add(key)
            points.append(rounded)
            break
        else:
            raise ValueError(
                "Unable to generate enough distinct initial Gaussian points "
                "inside the interior margins. Reduce n_points, margin or "
                "rounding, or widen the approved bounds."
            )
    return points



def _recompute_all_metrics(
    observations: list[Observation],
    extract_metrics: Callable[[Sequence[Any]], Sequence[float]],
    firstSet: bool = True,
) -> np.ndarray:
    """Recompute the metric for EVERY collected measurement.

    This is intentionally global: analyses such as PCA may change the metric
    assigned to old runs whenever a new run is added.

    Parameters
    ----------
    observations : list of Observation
        Measurements whose metrics must be refreshed.
    extract_metrics : callable
        Callback producing one metric per measurement.
    Returns
    -------
    numpy.ndarray
        The validated metric array in observation order.
    Raises
    ------
    ValueError
        If callback output has the wrong shape, length, or values.
    """
    measurements = [obs.measurement for obs in observations]
    metrics = np.asarray(extract_metrics(measurements, firstSet), dtype=float)

    if metrics.ndim != 1:
        raise ValueError(
            "extract_metrics(all_measurements) must return a 1-D sequence."
        )
    if len(metrics) != len(observations):
        raise ValueError(
            "extract_metrics(all_measurements) returned "
            f"{len(metrics)} metrics for {len(observations)} measurements. "
            "It must return exactly one metric per measurement, in the same order."
        )
    if not np.all(np.isfinite(metrics)):
        bad = np.where(~np.isfinite(metrics))[0].tolist()
        raise ValueError(
            f"extract_metrics returned non-finite values at indices {bad}."
        )

    for obs, metric in zip(observations, metrics):
        obs.metric = float(metric)

    return metrics

def estimate_noise_variance(
    observations: Sequence[Observation],
) -> float:
    """
    Estimate the common measurement-noise variance from repeated coordinates.

    If two independent measurements at the same coordinate are

        y1 = f(x) + e1
        y2 = f(x) + e2

    with Var(e)=sigma^2, then

        E[(y1-y2)^2] = 2*sigma^2.

    All same-coordinate pairs are pooled.

    This assumes all measurements have the same statistical exposure.

    Parameters
    ----------
    observations : sequence of Observation
        Observations containing repeated coordinates and metrics.
    Returns
    -------
    float
        Estimated common observation-noise variance.
    Raises
    ------
    RuntimeError
        If no coordinate has been measured more than once.
    """
    groups: dict[tuple[float, ...], list[float]] = {}

    for observation in observations:
        key = tuple(
            np.round(np.asarray(observation.coordinate, dtype=float), 12)
        )
        groups.setdefault(key, []).append(float(observation.metric))

    squared_differences = []

    for values in groups.values():
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                squared_differences.append(
                    (values[i] - values[j]) ** 2
                )

    if not squared_differences:
        raise RuntimeError(
            "Cannot estimate the absolute noise yet. "
            "Measure at least one coordinate twice."
        )

    noise_variance = float(np.mean(squared_differences) / 2.0)

    # Numerical floor only.  This is not an assumed physical noise level.
    y = np.asarray([obs.metric for obs in observations], dtype=float)
    scale = max(
        float(np.std(y)),
        float(np.ptp(y)) / 4.0 if len(y) > 1 else 0.0,
        1e-12,
    )
    floor = (1e-6 * scale) ** 2

    return max(noise_variance, floor)


def _fit_gp(
    observations: Sequence[Observation],
    bounds: np.ndarray,
    noise_variance: float,
    *,
    random_state: int,
) -> tuple[GaussianProcessRegressor, float, float, float]:
    """Fit a normalized Gaussian process to the current observations.

    Parameters
    ----------
    observations : sequence of Observation
        Observations with finite metrics.
    bounds : numpy.ndarray
        Physical coordinate bounds.
    noise_variance : float
        Estimated observation-noise variance.
    random_state : int
        Seed used by the Gaussian-process optimizer.
    Returns
    -------
    tuple[sklearn.gaussian_process.GaussianProcessRegressor, float, float, float]
        The fitted GP, metric offset, metric scale, and normalized noise variance.
    """
    X = np.vstack(
        [_to_unit(obs.coordinate, bounds) for obs in observations]
    )
    y = np.asarray([obs.metric for obs in observations], dtype=float)

    y_offset = float(np.mean(y))
    y_scale = float(np.std(y))

    if not np.isfinite(y_scale) or y_scale < 1e-12:
        y_scale = max(abs(y_offset), 1.0)

    y_normalized = (y - y_offset) / y_scale
    noise_variance_normalized = noise_variance / (y_scale ** 2)

    dimension = bounds.shape[0]

    kernel = ConstantKernel(
        1.0,
        constant_value_bounds=(1e-5, 1e3),
    ) * Matern(
        length_scale=np.full(dimension, 0.25),
        length_scale_bounds=(0.01, 5.0),
        nu=2.5,
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_variance_normalized,
        normalize_y=False,
        n_restarts_optimizer=3,
        random_state=int(random_state),
    )
    # A fitted kernel with bound-related ConvergenceWarnings is not treated as
    # trustworthy for selecting a KG point or reporting a model minimum.
    # Capture *any* fit warning, including constant_value and length_scale,
    # without hiding errors (which continue to propagate as exceptions).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.fit(X, y_normalized)
    if caught:
        raise GPFitWarning(
            [f"{warning.category.__name__}: {warning.message}" for warning in caught]
        )

    return (
        gp,
        y_offset,
        y_scale,
        noise_variance_normalized,
    )


def _posterior_covariance_with_candidate(
    gp: GaussianProcessRegressor,
    decision_points: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Return posterior covariance with a candidate measurement.

    Parameters
    ----------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        Fitted Gaussian process.
    decision_points : numpy.ndarray
        Unit-space points whose covariance is requested.
    candidate : numpy.ndarray
        Unit-space candidate measurement location.
    Returns
    -------
    numpy.ndarray
        Posterior covariance between each decision point and candidate.
    """
    candidate = np.atleast_2d(candidate)

    K_zx = gp.kernel_(decision_points, candidate).reshape(-1)
    K_zX = gp.kernel_(decision_points, gp.X_train_)
    K_Xx = gp.kernel_(gp.X_train_, candidate)

    Vz = solve_triangular(
        gp.L_,
        K_zX.T,
        lower=True,
        check_finite=False,
    )
    Vx = solve_triangular(
        gp.L_,
        K_Xx,
        lower=True,
        check_finite=False,
    )

    return K_zx - (Vz.T @ Vx).reshape(-1)


def gp_kg_next_coordinate(
    observations: Sequence[Observation],
    bounds: Sequence[Sequence[float]],
    *,
    n_candidate_points: int = 192,
    n_decision_points: int = 1024,
    quadrature_order: int = 15,
    random_state: int = 0,
) -> tuple[np.ndarray, float, float]:
    """
    Choose the next coordinate using an approximate correlated
    Knowledge Gradient (KG).

    Previously measured coordinates are included among the candidates, so the
    optimizer is allowed to repeat a point when replication is more useful
    than exploration.

    Parameters
    ----------
    observations:
        Existing observations, including at least one repeated coordinate.
    bounds:
        Physical lower and upper bounds for each dimension.
    n_candidate_points:
        Number of Sobol candidate points to evaluate.
    n_decision_points:
        Number of Sobol points approximating the final decision space.
    quadrature_order:
        Hermite quadrature order for the expected improvement estimate.
    random_state:
        Seed controlling Sobol sampling and GP fitting.

    Returns
    -------
    coordinate : ndarray
        Proposed next coordinate in physical parameter units.
    kg : float
        Expected one-step value of the measurement, in metric units.
    noise_std : float
        Current estimated standard deviation of one measurement.
    Raises
    ------
    ValueError
        If fewer than two observations are supplied.
    GPFitWarning
        If scikit-learn emits warnings during GP fitting; use
        ``next_coordinate_with_warning_recovery`` to acquire a random
        exploration point instead.
    """
    b = _validate_bounds(bounds)

    if len(observations) < 2:
        raise ValueError("At least two observations are required.")

    noise_variance = estimate_noise_variance(observations)
    noise_std = math.sqrt(noise_variance)

    gp, _, y_scale, noise_variance_n = _fit_gp(
        observations,
        b,
        noise_variance,
        random_state=random_state,
    )

    observed_unit = np.vstack(
        [_to_unit(obs.coordinate, b) for obs in observations]
    )

    dimension = b.shape[0]

    # Discrete representation of where the final minimum may lie.
    m_decision = int(
        np.ceil(np.log2(max(2, int(n_decision_points))))
    )
    decision_sampler = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=int(random_state),
    )
    decision_points = decision_sampler.random_base2(m=m_decision)
    decision_points = decision_points[: int(n_decision_points)]
    decision_points = np.vstack([decision_points, observed_unit])

    mean_decision = gp.predict(decision_points)
    current_minimum = float(np.min(mean_decision))

    # Candidate next measurement locations.
    m_candidate = int(
        np.ceil(np.log2(max(2, int(n_candidate_points))))
    )
    candidate_sampler = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=int(random_state) + 10000,
    )
    candidates = candidate_sampler.random_base2(m=m_candidate)
    candidates = candidates[: int(n_candidate_points)]

    # Always include the current posterior-mean optimum and all old points.
    current_best = decision_points[int(np.argmin(mean_decision))]
    candidates = np.vstack(
        [
            candidates,
            current_best[None, :],
            observed_unit,
        ]
    )

    nodes, weights = hermgauss(int(quadrature_order))
    z_nodes = np.sqrt(2.0) * nodes
    z_weights = weights / np.sqrt(np.pi)

    best_candidate = None
    best_kg_n = -np.inf

    for candidate in candidates:
        candidate_2d = candidate[None, :]

        k_xx = float(
            gp.kernel_(candidate_2d, candidate_2d)[0, 0]
        )
        K_Xx = gp.kernel_(gp.X_train_, candidate_2d)

        Vx = solve_triangular(
            gp.L_,
            K_Xx,
            lower=True,
            check_finite=False,
        )

        latent_variance = max(
            k_xx - float((Vx.T @ Vx)[0, 0]),
            0.0,
        )

        covariance = _posterior_covariance_with_candidate(
            gp,
            decision_points,
            candidate,
        )

        observation_variance = (
            latent_variance + noise_variance_n
        )

        if observation_variance <= 1e-15:
            kg_n = 0.0
        else:
            coefficient = (
                covariance / math.sqrt(observation_variance)
            )

            # For z~N(0,1), the posterior mean after the hypothetical
            # observation is:
            #
            #   mu_new = mu_old + coefficient * z
            possible_new_means = (
                mean_decision[:, None]
                + coefficient[:, None] * z_nodes[None, :]
            )

            possible_new_minima = np.min(
                possible_new_means,
                axis=0,
            )

            expected_new_minimum = float(
                np.dot(z_weights, possible_new_minima)
            )

            kg_n = max(
                current_minimum - expected_new_minimum,
                0.0,
            )

        if kg_n > best_kg_n:
            best_kg_n = kg_n
            best_candidate = candidate.copy()

    return (
        _from_unit(best_candidate, b),
        float(best_kg_n * y_scale),
        float(noise_std),
    )


def _random_exploration_coordinate(
    observations: Sequence[Observation],
    bounds: np.ndarray,
    minimum_step_sizes: np.ndarray,
    precisions: np.ndarray,
    *,
    random_state: int,
    max_attempts: int = 2048,
) -> np.ndarray:
    """Sample a new N-D point uniformly within physical bounds.

    Use rejection sampling after rounding, so random recovery avoids previously
    measured locations and obeys per-axis minimum movement constraints. Initial
    repeats for estimating noise are handled separately by the search loop.
    No arbitrary number of GP refits is attempted without new data.
    """
    rng = np.random.default_rng(int(random_state))
    seen = {tuple(np.asarray(obs.coordinate, dtype=float)) for obs in observations}
    previous = np.asarray(observations[-1].coordinate, dtype=float)

    for _ in range(max_attempts):
        x = _round_coordinate(rng.uniform(bounds[:, 0], bounds[:, 1]), precisions)
        x = np.clip(x, bounds[:, 0], bounds[:, 1])
        if tuple(x) in seen:
            continue
        difference = np.abs(x - previous)
        if np.any((difference > 0) & (difference + 1e-12 < minimum_step_sizes)):
            continue
        return x

    raise RandomExplorationExhausted(
        "Unable to sample an unmeasured random point satisfying the precision "
        "and minimum-step constraints. The discrete search space may be exhausted."
    )


def random_exploration_after_gp_warning(
    observations: Sequence[Observation],
    bounds: np.ndarray,
    minimum_step_sizes: np.ndarray,
    precisions: np.ndarray,
    warning_messages: Sequence[str],
    *,
    random_state: int,
) -> NextCoordinate:
    """Create an exploration proposal after any untrustworthy GP fit."""
    return NextCoordinate(
        coordinate=_random_exploration_coordinate(
            observations, bounds, minimum_step_sizes, precisions,
            random_state=int(random_state) + 1_000_000,
        ),
        kg=float("nan"),
        noise_std=math.sqrt(estimate_noise_variance(observations)),
        random_exploration=True,
        warning_messages=tuple(warning_messages),
    )


def next_coordinate_with_warning_recovery(
    observations: Sequence[Observation],
    bounds: np.ndarray,
    minimum_step_sizes: np.ndarray,
    precisions: np.ndarray,
    *,
    n_candidate_points: int = 192,
    n_decision_points: int = 1024,
    random_state: int = 0,
) -> NextCoordinate:
    """Use GP/KG normally; choose a random point when GP fitting warns.

    Every call refits the GP to all the available measurements. If it warns,
    the returned point is random and MUST be acquired before trying again.
    The calling loop's max_measurements is the upper limit on retries.
    ``kg`` is NaN while the GP is considered unreliable, so callers must not
    evaluate the KG convergence threshold during random exploration.
    """
    try:
        coordinate, kg, noise_std = gp_kg_next_coordinate(
            observations,
            bounds,
            n_candidate_points=n_candidate_points,
            n_decision_points=n_decision_points,
            random_state=random_state,
        )
    except GPFitWarning as problem:
        return random_exploration_after_gp_warning(
            observations, bounds, minimum_step_sizes, precisions,
            problem.messages, random_state=random_state,
        )

    return NextCoordinate(
        coordinate=_apply_minimum_step_sizes(
            coordinate, observations[-1].coordinate,
            minimum_step_sizes, precisions, bounds,
        ),
        kg=kg,
        noise_std=noise_std,
    )


def _model_minimum_or_best_measurement(
    observations: Sequence[Observation],
    bounds: np.ndarray,
    *,
    random_state: int,
    n_points: int = 8192,
) -> tuple[np.ndarray, float, bool, tuple[str, ...]]:
    """Never use an invalid GP fit to choose the final destination.

    If fitting warns at the budget limit, use the best actually measured
    point instead; return a flag so users can distinguish the measured
    fallback from a GP-predicted minimum.
    """
    try:
        x, y = _estimate_final_minimum(
            observations, bounds, random_state=random_state, n_points=n_points,
        )
    except GPFitWarning as problem:
        best = min(observations, key=lambda obs: obs.metric)
        return best.coordinate.copy(), float(best.metric), False, problem.messages
    return x, y, True, ()


def _validate_verbose(verbose: int | bool) -> int:
    """Normalise a verbosity level; keep bool callers backwards-compatible.

    0: no optional output, 1: measurements/best measured/warnings,
    2: level 1 plus predicted GP minimum and proposal diagnostics.
    ``True`` retains the former fully verbose behaviour (level 2).
    """
    if isinstance(verbose, bool):
        return 2 if verbose else 0
    if isinstance(verbose, int) and 0 <= verbose <= 2:
        return verbose
    raise ValueError("verbose must be 0, 1 or 2 (or False/True)")


def _print_best_measured(observations: Sequence[Observation]) -> None:
    """Report the best observed position without another GP fit."""
    best = min(observations, key=lambda obs: obs.metric)
    print(
        "     best measured minimum: "
        f"x={np.array2string(best.coordinate, precision=5)}, "
        f"metric={best.metric:.8g}"
    )


def _print_minimum_estimate(
    observations: Sequence[Observation],
    bounds: np.ndarray,
    *,
    random_state: int,
    n_points: int = 1024,
) -> None:
    """Print current best observation and the GP posterior-mean minimum.

    This is diagnostic only: the reported GP coordinate is *not* sent to a
    motor. Its predicted value is an approximation over Sobol points plus
    previously observed coordinates. The next acquisition is still chosen by
    KG, or randomly if fitting warns. If fitting warns, the only defensible
    estimate is the best measured coordinate.

    Fit with the same seed as the NEXT KG iteration. That way a bound warning
    here is consistent with warning-triggered exploration on the next step.
    """
    _print_best_measured(observations)
    try:
        x_est, value_est, fit_ok, _ = _model_minimum_or_best_measurement(
            observations, bounds, random_state=random_state, n_points=n_points,
        )
    except Exception as exc:
        # Reporting must not introduce a new failure mode for an experiment.
        # The optimiser itself will check its GP fit before selecting a point.
        print(
            "     estimated minimum: best measured (GP estimate unavailable: "
            f"{type(exc).__name__}: {exc})"
        )
        return

    if fit_ok:
        print(
            "     GP estimated minimum: "
            f"x={np.array2string(x_est, precision=5)}, "
            f"predicted_metric={value_est:.8g} "
            f"(posterior mean, {n_points} Sobol points + measurements)"
        )
    else:
        print(
            "     estimated minimum: best measured "
            "(GP fit warning; model prediction not trusted)"
        )


def _estimate_final_minimum(
    observations: Sequence[Observation],
    bounds: np.ndarray,
    *,
    random_state: int,
    n_points: int = 8192,
) -> tuple[np.ndarray, float]:
    """Estimate the final minimum from the fitted posterior mean.

    Parameters
    ----------
    observations : sequence of Observation
        Collected observations with finite metrics.
    bounds : numpy.ndarray
        Validated physical coordinate bounds.
    random_state : int
        Seed controlling the Sobol search.
    n_points : int, default=8192
        Number of unit-space points to evaluate.
    Returns
    -------
    tuple[numpy.ndarray, float]
        The estimated physical minimum coordinate and metric value.
    """
    noise_variance = estimate_noise_variance(observations)

    gp, y_offset, y_scale, _ = _fit_gp(
        observations,
        bounds,
        noise_variance,
        random_state=random_state,
    )

    dimension = bounds.shape[0]

    m = int(np.ceil(np.log2(max(2, int(n_points)))))
    sampler = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=int(random_state),
    )
    unit_points = sampler.random_base2(m=m)
    unit_points = unit_points[: int(n_points)]

    observed_unit = np.vstack(
        [_to_unit(obs.coordinate, bounds) for obs in observations]
    )
    unit_points = np.vstack([unit_points, observed_unit])

    posterior_mean = (
        gp.predict(unit_points) * y_scale
        + y_offset
    )

    index = int(np.argmin(posterior_mean))

    return (
        _from_unit(unit_points[index], bounds),
        float(posterior_mean[index]),
    )


def search_minimum(
    accuracy: float,
    bounds: Sequence[Sequence[float]],
    get_measurement: Callable[[np.ndarray], Any],
    extract_metrics: Callable[[Sequence[Any]], Sequence[float]],
    *,
    initial_coordinates: Optional[Iterable[Sequence[float]]] = None,
    initial_repeats: int = 2,
    max_measurements: int = 30,
    minimum_step_sizes: Optional[Sequence[float]] = None,
    precisions: Optional[Sequence[int]] = None,
    random_state: int = 12345,
    n_candidate_points: int = 192,
    n_decision_points: int = 1024,
    progress_callback: Optional[
        Callable[[Sequence[Observation], Optional[float]], None]
    ] = None,
    verbose: int | bool = 2,
    minimum_estimate_points: int = 1024,
) -> SearchResult:
    """Sequentially search for a noisy minimum.

    Parameters
    ----------
    accuracy:
        Stop when the current one-step Knowledge Gradient is below this value.
        Units are the CURRENT units returned by extract_metrics().

    bounds:
        [(min_1,max_1), ..., (min_N,max_N)].

    get_measurement:
        User callback::

            measurement = get_measurement(coordinate)

    extract_metrics:
        Global user callback::

            metrics = extract_metrics(all_measurements)

        It receives EVERY measurement collected so far and must return one
        scalar metric per measurement, in the same order. It is re-run after
        every new sequential measurement, so earlier metrics may change.
        This is suitable for PCA and other global analyses.

    initial_coordinates:
        User-specified starting coordinates. They do NOT need to be extrema.
        If None, default_initial_coordinates(bounds) is used only as a
        convenience fallback.

    minimum_step_sizes:
        Minimum per-dimension change between successive non-repeated
        measurements. A proposed change smaller than the configured value is
        expanded in the proposed direction, subject to the bounds. Repeated
        initial measurements are preserved for noise estimation.

    precisions:
        Number of decimal places allowed for each coordinate. Proposed,
        measured, reported, and final coordinates are rounded accordingly.

    initial_repeats:
        Number of measurements at each initial coordinate. Normally >=2 so
        the unknown absolute noise can be estimated.

    random_state:
        Seed for reproducible GP proposals and warning-triggered random
        exploration. Each warned fit adds one new random measurement and
        triggers a fresh fit at the next iteration, up to max_measurements.
        Warning-triggered acquisitions do not count as convergence.

    progress_callback:
        Optional callback after all current metrics have been recomputed::

            progress_callback(observations, estimated_noise_std)

        Every Observation.metric is the CURRENT globally recomputed value.

    verbose:
        Output level: 0 = quiet, 1 = measurements, best measured minimum and
        fit warnings, 2 = level 1 plus GP-predicted minimum and proposal
        diagnostics. True/False remain accepted as aliases for 2/0. Only
        level 2 runs an extra GP fit for progress reporting.
    minimum_estimate_points:
        Number of Sobol points used for the progress-only GP estimate. More
        points give a finer approximation but require extra GP fits/time.

    Returns
    -------
    SearchResult
        The estimated minimum and complete observation history.
    Raises
    ------
    ValueError
        If search settings or callback results are invalid.
    RuntimeError
        If no measurements are acquired.
    """
    b = _validate_bounds(bounds)
    verbose = _validate_verbose(verbose)
    minimum_steps = _validate_minimum_step_sizes(
        minimum_step_sizes,
        b.shape[0],
    )
    coordinate_precisions = _validate_precisions(precisions, b.shape[0])

    if accuracy <= 0:
        raise ValueError("accuracy must be > 0.")
    if initial_repeats < 2:
        raise ValueError(
            "initial_repeats must be >= 2 when absolute noise is unknown."
        )
    if max_measurements < 2:
        raise ValueError("max_measurements must be >= 2.")
    if (isinstance(minimum_estimate_points, bool)
            or not isinstance(minimum_estimate_points, int)
            or not 2 <= minimum_estimate_points <= 65536):
        raise ValueError("minimum_estimate_points must be an integer from 2 to 65536")

    if initial_coordinates is None:
        initial_coordinates = default_initial_coordinates(b)

    initial_coordinates = [
        _round_coordinate(_validate_coordinate(x, b), coordinate_precisions, b)
        for x in initial_coordinates
    ]
    if not initial_coordinates:
        raise ValueError("At least one initial coordinate is required.")

    observations: list[Observation] = []
    converged = False
    kg = float("nan")
    random_explorations = 0
    gp_warnings: list[str] = []

    def acquire(coordinate: np.ndarray) -> None:
        """Acquire one measurement and append it to the observation list.

        Parameters
        ----------
        coordinate : numpy.ndarray
            Coordinate to pass to the measurement callback.
        """
        measurement = get_measurement(coordinate.copy())
        observations.append(
            Observation(
                coordinate=coordinate.copy(),
                measurement=measurement,
            )
        )

    def update_metrics_and_report(firstSet: bool = False) -> Optional[float]:
        """Recompute metrics, notify the callback, and optionally log progress.

        Returns
        -------
        float or None
            The current estimated noise standard deviation, if estimable.
        """
        _recompute_all_metrics(observations, extract_metrics, firstSet)

        noise_std = None
        try:
            noise_std = math.sqrt(estimate_noise_variance(observations))
        except RuntimeError:
            pass

        if progress_callback is not None:
            progress_callback(observations, noise_std)

        if verbose:
            latest = observations[-1]
            message = (
                f"{len(observations):3d}: "
                f"x={np.array2string(latest.coordinate, precision=6)}, "
                f"current_metric={latest.metric:.8g}"
            )
            if noise_std is not None:
                message += f", estimated_noise_sigma={noise_std:.6g}"
            print(message)
            if noise_std is not None:
                if verbose >= 2:
                    _print_minimum_estimate(
                        observations, b,
                        random_state=random_state + len(observations),
                        n_points=minimum_estimate_points,
                    )
                else:
                    _print_best_measured(observations)

        return noise_std

    # Collect the whole initial design before the first global metric fit.
    # This is friendlier to PCA and similar methods that need multiple runs.
    # Progress is still shown for each initial acquisition, with the estimate
    # clearly marked as pending until PCA/noise estimates are available.
    n_initial = min(max_measurements, len(initial_coordinates) + initial_repeats - 1)
    for index, coordinate in enumerate(initial_coordinates):
        # only repeat the first coordinate to estimate noise variance
        repeats = initial_repeats if index == 0 else 1
        for _ in range(repeats):
            if len(observations) >= max_measurements:
                break
            acquire(coordinate)
            if verbose and len(observations) < n_initial:
                print(
                    f"{len(observations):3d}: "
                    f"x={np.array2string(coordinate, precision=6)}, "
                    "estimated minimum pending (collecting initial design)"
                )
        if len(observations) >= max_measurements:
            break

    if not observations:
        raise RuntimeError("No measurements were acquired.")

    firstSet = True
    update_metrics_and_report(firstSet)

    # Also verifies that repeated coordinates exist.
    estimate_noise_variance(observations)

    while len(observations) < max_measurements:
        try:
            proposal = next_coordinate_with_warning_recovery(
                observations, b, minimum_steps, coordinate_precisions,
                n_candidate_points=n_candidate_points,
                n_decision_points=n_decision_points,
                random_state=random_state + len(observations),
            )
        except RandomExplorationExhausted as exc:
            # No valid *new* random point remains on the configured grid.
            if verbose:
                print(f"     Random exploration stopped: {exc}")
            break

        # A clean KG fit can still be followed by a warned *final* model fit
        # (the final fit uses its own random seed). Do not declare convergence
        # while there are fit warnings and unused measurement slots remain.
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
                    observations, b, minimum_steps, coordinate_precisions,
                    fit_warnings, random_state=random_state + len(observations),
                )
            except RandomExplorationExhausted:
                if verbose:
                    print("     No new random coordinate after final-model fit warning")
                break

        coordinate, kg, noise_std = proposal.coordinate, proposal.kg, proposal.noise_std
        if proposal.random_exploration:
            random_explorations += 1
            gp_warnings.extend(proposal.warning_messages)
            if verbose:
                for message in proposal.warning_messages:
                    print(f"     GP fit warning -> random exploration: {message}")
        if verbose >= 2:
            print(
                "     next="
                f"{np.array2string(coordinate, precision=6)}, "
                f"strategy={'random' if proposal.random_exploration else 'KG'}, "
                f"KG={kg:.6g}, noise_sigma={noise_std:.6g}"
            )

        acquire(coordinate)

        # Critical for PCA/global metrics: every old metric is recomputed too.
        update_metrics_and_report()

        # Verbosity 2 optionally fits a diagnostic GP in the progress callback.
        # Other levels avoid that extra calculation; proposal selection remains
        # in the next iteration for all verbosity levels.

    # One final global recomputation ensures the returned state is consistent.
    _recompute_all_metrics(observations, extract_metrics)

    final_coordinate, final_value, model_fit_ok, final_warnings = _model_minimum_or_best_measurement(
        observations, b, random_state=random_state + 99999,
    )
    gp_warnings.extend(final_warnings)
    if final_warnings and verbose:
        print("     Final GP fit still warns: returning the best measured point")
    final_coordinate = _round_coordinate(
        final_coordinate,
        coordinate_precisions,
        b,
    )

    noise_std = math.sqrt(estimate_noise_variance(observations))

    return SearchResult(
        coordinate=final_coordinate,
        value=final_value,
        estimated_noise_std=noise_std,
        n_measurements=len(observations),
        converged=converged and model_fit_ok,
        observations=observations,
        kg=kg,
        random_explorations=random_explorations,
        model_fit_ok=model_fit_ok,
        gp_warnings=tuple(gp_warnings),
    )
