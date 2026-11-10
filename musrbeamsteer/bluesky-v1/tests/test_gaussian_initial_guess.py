"""Gaussian operator-centred initial sampling without hardware access."""
from __future__ import annotations

import numpy as np
import pytest

from ..blueskysteercore.gp_kg_minimizer import gaussian_initial_coordinates
from examples import musr_steering_main as example

BOUNDS = ((-0.5, 0.5), (-1.5, 1.5))


def test_real_musr_example_configuration_matches_operator_values():
    assert example.USE_SYNTHETIC_DATA
    assert example.USE_SYNTHETIC_MOTORS
    steering = example.PV_DF[example.PV_DF["type"] == "steering"]
    assert steering["setpoint_pv"].tolist() == [
        "IN:MUSR:KEPCO_04:CURRENT:SP",
        "IN:MUSR:KEPCO_05:CURRENT:SP",
    ]
    assert steering["readback_pv"].tolist() == [
        "IN:MUSR:KEPCO_04:CURRENT",
        "IN:MUSR:KEPCO_05:CURRENT",
    ]
    assert steering["safe_bounds"].tolist() == [(-0.5, 0.5), (-1.5, 1.5)]
    assert steering["tolerance"].tolist() == [0.01, 0.01]
    assert steering["precision"].tolist() == [3, 3]
    assert example.CENTER == [0.1, -0.3, 0.0, 0.0, 60.0]
    assert example.CURVATURE == [3.0, 3.0, 1.0, 1.0, 0.001]
    assert example.METRIC_TIME_WINDOW_NS == (0., 30000.)
    assert example.METRIC_REBIN == 1
    assert example.MAX_MEASUREMENTS == 50
    assert example.VERBOSE == 2
    assert example.DURATION.unit.value == 'events'
    assert example.DURATION.value == 1_000_000
def test_reproducible_gaussian_design_avoids_boundary_and_quantises():
    points = gaussian_initial_coordinates(BOUNDS, [0.1, -0.3],
        edge_margin_fraction=.02, precisions=[3, 3], random_state=20261023)
    copy = gaussian_initial_coordinates(BOUNDS, [0.1, -0.3],
        edge_margin_fraction=.02, precisions=[3, 3], random_state=20261023)
    assert len(points) == 5  # 1 + 2*N, first is repeated by the optimiser
    assert len({tuple(x) for x in points}) == len(points)
    assert np.allclose(points, copy)
    assert any(not np.allclose(p, [0.1, -.3]) for p in points)
    # 2% of the full span must be excluded at each side.
    assert all(-.48 < x[0] < .48 and -1.44 < x[1] < 1.44 for x in points)
    assert all(np.allclose(x * 1000, np.round(x * 1000)) for x in points)


def test_n_dimensional_samples_follow_nearest_edge_over_three():
    # Non-symmetric guess means each axis has a distinct standard deviation.
    bounds = [(0., 6.), (-3., 7.), (-2., 2.)]
    guess = [1., 5., 0.5]
    sigma = np.minimum(np.asarray(guess) - np.asarray(bounds)[:, 0],
                       np.asarray(bounds)[:, 1] - np.asarray(guess)) / 3
    assert np.allclose(sigma, [1/3, 2/3, 1.5/3])
    points = np.asarray(gaussian_initial_coordinates(bounds, guess,
        n_points=2000, edge_margin_fraction=0, random_state=123))
    assert points.shape == (2000, 3)
    # The extreme tails are truncated by the bounds: tolerate some decrease.
    assert np.allclose(points.mean(axis=0), guess, atol=.08)
    assert np.allclose(points.std(axis=0), sigma, rtol=.15)
    assert np.all(points > np.asarray(bounds)[:, 0])
    assert np.all(points < np.asarray(bounds)[:, 1])


@pytest.mark.parametrize('guess', [
    [0.5, 0.0],  # boundary
    [0.49, 0.0],  # in restricted 2% edge margin
    [0.0, 1.49],
    [0.0],
    [np.nan, 0.0],
    [np.inf, 0.0],
])
def test_reject_unsafe_initial_guess(guess):
    with pytest.raises(ValueError):
        gaussian_initial_coordinates(BOUNDS, guess)


@pytest.mark.parametrize('extra', [
    {'edge_margin_fraction': .5},
    {'edge_margin_fraction': -0.1},
    {'n_points': 0},
    {'n_points': True},
    {'precisions': [0]},
])
def test_invalid_sampling_arguments(extra):
    with pytest.raises(ValueError):
        gaussian_initial_coordinates(BOUNDS, [0, 0], **extra)


def test_insufficient_discrete_points_never_clips_to_edge():
    with pytest.raises(ValueError, match='Unable to generate enough'):
        gaussian_initial_coordinates([(-1.5, 1.5)], [0.0],
                                     n_points=7, precisions=[0],
                                     max_attempts_per_point=100)
