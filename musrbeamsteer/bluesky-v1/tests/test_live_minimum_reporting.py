"""Live GP minimum reporting, without EPICS or a hardware RunEngine."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ..blueskysteercore import gp_kg_minimizer as gp
from ..blueskysteercore import bluesky_integration as integration
from .test_bluesky_integration import FakeMotor, fake_stubs, consume


def install_fake_bluesky(monkeypatch):
    bluesky = types.ModuleType("bluesky")
    bluesky.__path__ = []
    stubs = fake_stubs()
    bluesky.plan_stubs = stubs
    monkeypatch.setitem(sys.modules, "bluesky", bluesky)
    monkeypatch.setitem(sys.modules, "bluesky.plan_stubs", stubs)


def objective_records(records, first_set=False):
    return [record["metric"] for record in records]


def fake_acquire(positions):
    def acquire(x):
        positions.append(np.asarray(x).copy())
        if False:
            yield
        return {"metric": (float(x[0]) - 0.3) ** 2 + 0.002 * len(positions)}
    return acquire


def set_simple_gp(monkeypatch, *, warned_at=()):
    fit_calls = []

    def model_estimate(observations, bounds, *, random_state, n_points=8192):
        fit_calls.append((len(observations), n_points, random_state))
        if len(observations) in warned_at:
            best = min(observations, key=lambda item: item.metric)
            return best.coordinate.copy(), best.metric, False, ("ConvergenceWarning: length_scale",)
        return np.array([0.31]), 0.0123, True, ()

    # Both modules import the model helper; patch both call sites.
    monkeypatch.setattr(gp, "_model_minimum_or_best_measurement", model_estimate)
    monkeypatch.setattr(integration, "_model_minimum_or_best_measurement", model_estimate)
    monkeypatch.setattr(gp, "gp_kg_next_coordinate",
                        lambda *args, **kwargs: (np.array([0.33]), 1.0, 0.03))
    return fit_calls


def test_bluesky_reports_best_and_model_after_every_sequential_measurement(monkeypatch, capsys):
    install_fake_bluesky(monkeypatch)
    fit_calls = set_simple_gp(monkeypatch)
    motor = FakeMotor(0.0)
    positions = []
    result = consume(integration.minimize_plan(
        [motor], [(-1.0, 1.0)], fake_acquire(positions), objective_records,
        initial_coordinates=[(0.0,), (0.5,)], initial_repeats=2,
        max_measurements=5, final_position="last",
        minimum_estimate_points=64, verbose=True,
    ))
    out = capsys.readouterr().out
    assert "  1: x=[0.]" in out
    assert "estimated minimum pending (collecting initial design)" in out
    # The model becomes available after the full initial design at measurement 3.
    assert out.count("best measured minimum:") == 3
    assert out.count("GP estimated minimum:") == 3
    assert "x=[0.31], predicted_metric=0.0123" in out
    assert [count for count, points, seed in fit_calls].count(3) == 1
    assert [points for count, points, seed in fit_calls if points == 64] == [64, 64, 64]
    assert result.n_measurements == len(positions) == 5
    assert motor.value == pytest.approx(float(positions[-1][0]))


def test_warned_model_never_reports_untrusted_prediction(monkeypatch, capsys):
    install_fake_bluesky(monkeypatch)
    set_simple_gp(monkeypatch, warned_at=(3,))
    positions = []
    consume(integration.minimize_plan(
        [FakeMotor(0.0)], [(-1, 1)], fake_acquire(positions), objective_records,
        initial_coordinates=[(0.0,), (0.5,)], initial_repeats=2,
        max_measurements=4, final_position="last", verbose=True,
    ))
    out = capsys.readouterr().out
    assert "estimated minimum: best measured (GP fit warning; model prediction not trusted)" in out
    assert out.count("GP estimated minimum:") == 1
    assert out.count("best measured minimum:") == 2
    assert len(positions) == 4


def test_sync_minimum_estimates_after_each_new_point(monkeypatch, capsys):
    fit_calls = set_simple_gp(monkeypatch)
    measured = []

    def get_measurement(x):
        measured.append(np.asarray(x).copy())
        return {"metric": (float(x[0]) - 0.3) ** 2 + len(measured) * .0001}

    gp.search_minimum(
        1e-4, [(-1, 1)], get_measurement, objective_records,
        initial_coordinates=[(0.0,), (0.5,)], initial_repeats=2,
        max_measurements=5, minimum_estimate_points=32, verbose=True,
    )
    out = capsys.readouterr().out
    assert out.count("GP estimated minimum:") == 3
    assert [points for count, points, seed in fit_calls if points == 32] == [32, 32, 32]
    assert len(measured) == 5


def test_silent_bluesky_does_not_fit_extra_progress_models(monkeypatch):
    install_fake_bluesky(monkeypatch)
    fit_calls = set_simple_gp(monkeypatch)
    positions = []
    consume(integration.minimize_plan(
        [FakeMotor(0.0)], [(-1, 1)], fake_acquire(positions), objective_records,
        initial_coordinates=[(0.0,), (0.5,)], initial_repeats=2,
        max_measurements=4, final_position="last", verbose=False,
    ))
    assert len(positions) == 4
    # Only the required final model evaluation, not once per acquisition.
    assert len(fit_calls) == 1
    assert fit_calls[0][1] == 8192


def test_rejects_invalid_progress_resolution_without_hardware_moves(monkeypatch):
    install_fake_bluesky(monkeypatch)
    def fail_acquire(x):
        raise AssertionError("acquisition must not happen")
        yield
    for n in (0, 1, -1, 1.5, True, 65537):
        with pytest.raises(ValueError, match="minimum_estimate_points"):
            consume(integration.minimize_plan(
                [FakeMotor(0.0)], [(-1, 1)], fail_acquire, objective_records,
                minimum_estimate_points=n, verbose=False,
            ))


def test_bluesky_verbose_level_one_reports_best_without_extra_gp_fits(monkeypatch, capsys):
    install_fake_bluesky(monkeypatch)
    fit_calls = set_simple_gp(monkeypatch)
    positions = []
    result = consume(integration.minimize_plan(
        [FakeMotor(0.0)], [(-1.0, 1.0)], fake_acquire(positions), objective_records,
        initial_coordinates=[(0.0,), (0.5,)], initial_repeats=2,
        max_measurements=5, final_position="last", verbose=1,
    ))
    out = capsys.readouterr().out
    assert "best measured minimum:" in out
    assert out.count("best measured minimum:") == 3
    assert "GP estimated minimum:" not in out
    assert "next=" not in out
    assert "estimated minimum pending" in out
    assert [n for _, n, _ in fit_calls] == [8192]
    assert result.n_measurements == 5


def test_sync_verbose_level_one_reports_best_without_extra_gp_fits(monkeypatch, capsys):
    fit_calls = set_simple_gp(monkeypatch)
    records = []

    def get_measurement(x):
        records.append(x.copy())
        return {"metric": float((x[0] - 0.3) ** 2 + .001 * len(records))}

    result = gp.search_minimum(
        1e-4, [(-1, 1)], get_measurement, objective_records,
        initial_coordinates=[(0.0,), (0.5,)], initial_repeats=2,
        max_measurements=5, verbose=1,
    )
    out = capsys.readouterr().out
    assert out.count("best measured minimum:") == 3
    assert "GP estimated minimum:" not in out
    assert "next=" not in out
    assert [n for _, n, _ in fit_calls] == [8192]
    assert result.n_measurements == len(records) == 5


def test_numeric_level_two_retains_gp_diagnostics(monkeypatch, capsys):
    install_fake_bluesky(monkeypatch)
    fit_calls = set_simple_gp(monkeypatch)
    positions = []
    consume(integration.minimize_plan(
        [FakeMotor(0.0)], [(-1, 1)], fake_acquire(positions), objective_records,
        initial_coordinates=[(0.0,), (0.5,)], initial_repeats=2,
        max_measurements=4, final_position="last", verbose=2,
        minimum_estimate_points=16,
    ))
    out = capsys.readouterr().out
    assert "GP estimated minimum:" in out
    assert "next=" in out
    assert [n for _, n, _ in fit_calls if n == 16] == [16, 16]


def test_verbose_levels_are_validated_before_move(monkeypatch):
    install_fake_bluesky(monkeypatch)
    for bad in (-1, 3, 1.5, None, "full"):
        with pytest.raises(ValueError, match="verbose must be"):
            consume(integration.minimize_plan(
                [FakeMotor(0.0)], [(-1, 1)], fake_acquire([]), objective_records,
                verbose=bad,
            ))
        with pytest.raises(ValueError, match="verbose must be"):
            gp.search_minimum(1e-4, [(-1, 1)], lambda x: None,
                              objective_records, verbose=bad)
    assert gp._validate_verbose(False) == 0
    assert gp._validate_verbose(True) == 2
