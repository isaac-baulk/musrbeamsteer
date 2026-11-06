"""Gaussian operator-centred initial sampling without hardware access."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ..blueskysteercore.gp_kg_minimizer import gaussian_initial_coordinates
from examples import musr_steering_main as example

BOUNDS = ((-0.5, 0.5), (-1.5, 1.5))


def test_real_musr_example_configuration_matches_operator_values():
    assert example.USE_SYNTHETIC_DATA
    assert example.USE_SYNTHETIC_MOTORS
    assert example.STEER_1_SETPOINT_PV == 'IN:MUSR:KEPCO_04:CURRENT:SP'
    assert example.STEER_1_READBACK_PV == 'IN:MUSR:KEPCO_04:CURRENT'
    assert example.STEER_2_SETPOINT_PV == 'IN:MUSR:KEPCO_05:CURRENT:SP'
    assert example.STEER_2_READBACK_PV == 'IN:MUSR:KEPCO_05:CURRENT'
    assert example.SAFE_BOUNDS == BOUNDS
    assert example.READBACK_TOLERANCES == (0.01, 0.01)
    assert example.CURRENT_PRECISIONS == [3, 3]
    assert example.CENTER == [0.1, -0.3]
    assert example.CURVATURE == [3.0, 3.0]
    assert example.METRIC_NAME == 'pca'
    assert example.METRIC_TIME_WINDOW_NS == (0., 30000.)
    assert example.METRIC_REBIN == 1
    assert example.MAX_MEASUREMENTS == 50
    assert example.VERBOSE == 2
    assert example.DURATION.unit.value == 'events'
    assert example.DURATION.value == 1_000_000
    assert example.INITIAL_GUESS == [0, 0]


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


def test_prompt_reprompts_and_parses_list_or_plain_numbers(monkeypatch):
    monkeypatch.setattr(example, 'INITIAL_GUESS', None)
    entries = iter(['[0.5, 0.0]', '[0.0]', 'oops', '0.10, -0.30'])
    prompts = []

    def fake_input(message):
        prompts.append(message)
        return next(entries)

    guess = example.resolve_initial_guess(input_fn=fake_input)
    assert guess == [0.10, -0.30]
    assert len(prompts) == 4
    assert '2 steering currents' in prompts[0]
    assert example._parse_initial_guess('[0, -0.3]', 2) == [0, -0.3]
    assert example._parse_initial_guess('0 -0.3', 2) == [0, -0.3]


def test_configured_guess_skips_input_and_invalid_guess_fails(monkeypatch):
    def forbidden_input(prompt):
        raise AssertionError('No prompt should be required')
    monkeypatch.setattr(example, 'INITIAL_GUESS', [0.1, -0.3])
    assert example.resolve_initial_guess(input_fn=forbidden_input) == [0.1, -.3]
    monkeypatch.setattr(example, 'INITIAL_GUESS', [0.5, -0.3])
    with pytest.raises(ValueError, match='Configured INITIAL_GUESS'):
        example.resolve_initial_guess(input_fn=forbidden_input)


def test_config_prompt_and_initial_design_happen_before_any_epics_connection(monkeypatch):
    monkeypatch.setattr(example, 'INITIAL_GUESS', [0., -.2])
    monkeypatch.setattr(example, 'VERBOSE', 0)
    monkeypatch.setattr(example, 'USE_SYNTHETIC_DATA', True)
    monkeypatch.setattr(example, 'USE_SYNTHETIC_MOTORS', False)
    monkeypatch.setattr(example, 'PLOT_LIVE', False)
    calls = []
    fake_module = types.ModuleType('ophyd_async')
    fake_module.__path__ = []
    plans = types.ModuleType('ophyd_async.plan_stubs')

    def ensure_connected(*devices, timeout):
        calls.append('connect')
        if False:
            yield

    plans.ensure_connected = ensure_connected
    monkeypatch.setitem(sys.modules, 'ophyd_async', fake_module)
    monkeypatch.setitem(sys.modules, 'ophyd_async.plan_stubs', plans)
    bluesky = types.ModuleType('bluesky')
    bluesky.__path__ = []
    bluesky_plan_stubs = types.ModuleType('bluesky.plan_stubs')
    monkeypatch.setitem(sys.modules, 'bluesky', bluesky)
    monkeypatch.setitem(sys.modules, 'bluesky.plan_stubs', bluesky_plan_stubs)

    monkeypatch.setattr(example, 'make_setpoint_readback_axis', lambda **kwargs: object())

    def minimize_plan(*args, **kwargs):
        calls.append('minimise')
        assert len(kwargs['initial_coordinates']) == 5
        assert all(-.48 < x[0] < .48 and -1.44 < x[1] < 1.44
                   for x in kwargs['initial_coordinates'])
        assert kwargs['initial_repeats'] == 2
        if False:
            yield
        return 'ok'

    monkeypatch.setattr(example, 'minimize_plan', minimize_plan)
    generator = example.make_plan()
    assert calls == []  # no hardware access until RunEngine executes the plan
    with pytest.raises(StopIteration) as ended:
        next(generator)
    assert ended.value.value == 'ok'
    assert calls == ['connect', 'minimise']
