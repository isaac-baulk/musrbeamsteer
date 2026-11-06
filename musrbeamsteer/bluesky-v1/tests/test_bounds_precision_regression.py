"""Regression tests for half-integer bounds on live steering magnets."""
import sys
import types

import numpy as np
import pytest

from ..blueskysteercore import gp_kg_minimizer as gp
from ..blueskysteercore import bluesky_integration as integration
from .test_bluesky_integration import FakeMotor, fake_stubs, consume


def test_none_precision_keeps_fractional_currents():
    bounds = gp._validate_bounds([(-1.5, 1.5), (-1.5, 1.5)])
    digits = gp._validate_precisions(None, 2)
    assert np.allclose(gp._round_coordinate([-1.25, 0.123456], digits, bounds),
                       [-1.25, 0.123456])
    assert np.allclose(
        gp._apply_minimum_step_sizes([0.023, 0.012], [0.0, 0.0],
                                     np.asarray([0.05, 0.02]), digits, bounds),
        [0.05, 0.02],
    )


def test_integer_precision_clips_rounded_half_integer_bounds():
    bounds = gp._validate_bounds([(-1.5, 1.5)])
    digits = gp._validate_precisions([0], 1)
    assert np.allclose(gp._round_coordinate([-1.5], digits, bounds), [-1.5])
    assert np.allclose(gp._round_coordinate([1.5], digits, bounds), [1.5])


def test_bluesky_plan_never_moves_outside_approved_current_range(monkeypatch):
    bluesky = types.ModuleType('bluesky')
    bluesky.__path__ = []
    stubs = fake_stubs()
    bluesky.plan_stubs = stubs
    monkeypatch.setitem(sys.modules, 'bluesky', bluesky)
    monkeypatch.setitem(sys.modules, 'bluesky.plan_stubs', stubs)
    monkeypatch.setattr(integration, '_model_minimum_or_best_measurement',
                        lambda *args, **kwargs: (np.array([0.0]), 0.0, True, ()))
    points = []

    def acquire(x):
        points.append(x.copy())
        if False:
            yield
        return {'metric': float(x[0] ** 2) + 0.001 * len(points)}

    def extract(records, first_set=False):
        return [record['metric'] for record in records]

    result = consume(integration.minimize_plan(
        [FakeMotor(0.0)], [(-1.5, 1.5)], acquire, extract,
        max_measurements=4, initial_repeats=2,
        precisions=[0], verbose=False,
    ))
    assert result.n_measurements == 4
    assert np.allclose([p[0] for p in points], [0, 0, -1.5, 1.5])


def test_aborts_before_move_if_initial_readback_outside_bounds(monkeypatch):
    bluesky = types.ModuleType('bluesky')
    bluesky.__path__ = []
    stubs = fake_stubs()
    bluesky.plan_stubs = stubs
    monkeypatch.setitem(sys.modules, 'bluesky', bluesky)
    monkeypatch.setitem(sys.modules, 'bluesky.plan_stubs', stubs)

    motor = FakeMotor(-2.0)

    def acquire(x):
        raise AssertionError('Acquisition should not run')
        yield

    with pytest.raises(ValueError, match='outside bounds'):
        consume(integration.minimize_plan([motor], [(-1.5, 1.5)], acquire,
                                          lambda records, first_set=False: [],
                                          max_measurements=3, verbose=False))
    assert motor.value == -2.0
