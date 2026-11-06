"""GP convergence warning recovery (no real hardware required)."""
import warnings
import numpy as np
import pytest

from ..blueskysteercore import gp_kg_minimizer as gp
from ..blueskysteercore import bluesky_integration as integration
from .test_bluesky_integration import FakeMotor, fake_stubs, consume


def install_fake_bluesky(monkeypatch):
    import sys
    import types

    bluesky = types.ModuleType('bluesky')
    bluesky.__path__ = []
    stubs = fake_stubs()
    bluesky.plan_stubs = stubs
    monkeypatch.setitem(sys.modules, 'bluesky', bluesky)
    monkeypatch.setitem(sys.modules, 'bluesky.plan_stubs', stubs)


def make_acquisition(seen):
    def acquire(x):
        seen.append(np.array(x).copy())
        if False:
            yield
        return {'metric': float(np.sum(x ** 2)) + 0.001 * len(seen)}
    return acquire


def simple_metrics(measurements, first_set=False):
    return [m['metric'] for m in measurements]


def test_sklearn_fit_warning_is_detected(monkeypatch):
    """Both typical kernel warnings are captured and treated as untrusted."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.exceptions import ConvergenceWarning

    def warning_fit(self, *args, **kwargs):
        warnings.warn('The optimal value of constant_value is close to the upper bound', ConvergenceWarning)
        warnings.warn('The optimal value of length_scale is close to the lower bound', ConvergenceWarning)
        return self

    monkeypatch.setattr(GaussianProcessRegressor, 'fit', warning_fit)
    observations = [gp.Observation(np.array([0.0]), metric=1.0),
                    gp.Observation(np.array([1.0]), metric=2.0)]
    with pytest.raises(gp.GPFitWarning) as caught:
        gp._fit_gp(observations, np.array([[-1.0, 1.0]]), 0.1, random_state=12)
    assert 'constant_value' in str(caught.value)
    assert 'length_scale' in str(caught.value)
    assert len(caught.value.messages) == 2


def test_bluesky_random_until_gp_recovers(monkeypatch):
    install_fake_bluesky(monkeypatch)
    motors = [FakeMotor(0.0), FakeMotor(0.0)]
    seen = []
    fit_sizes = []

    def fake_kg(observations, bounds, **kw):
        fit_sizes.append(len(observations))
        if len(observations) < 5:
            raise gp.GPFitWarning(['ConvergenceWarning: length_scale at bound'])
        return np.array([0.4, -0.5]), 1.0, 0.03

    monkeypatch.setattr(gp, 'gp_kg_next_coordinate', fake_kg)
    monkeypatch.setattr(integration, '_model_minimum_or_best_measurement',
                        lambda *args, **kwargs: (np.array([0.4, -0.5]), 0.0, True, ()))
    result = consume(integration.minimize_plan(
        motors, [(-2.0, 2.0), (-3.0, 3.0)], make_acquisition(seen), simple_metrics,
        initial_coordinates=[(0.0, 0.0), (1.0, 1.0)],
        initial_repeats=2, max_measurements=6,
        minimum_step_sizes=[0.1, 0.2], precisions=[2, 2],
        random_state=5, final_position='last', verbose=False,
    ))
    assert fit_sizes == [3, 4, 5]
    assert result.random_explorations == 2
    assert result.n_measurements == 6
    assert result.model_fit_ok
    assert not result.converged
    assert len(result.gp_warnings) == 2
    assert np.allclose(seen[-1], [0.4, -0.5])  # KG restored after recovery
    for i in [3, 4]:
        assert np.all(seen[i] >= [-2, -3]) and np.all(seen[i] <= [2, 3])
        assert np.allclose(seen[i] * 100, np.round(seen[i] * 100))
        assert tuple(seen[i]) not in [tuple(p) for p in seen[:i]]
        delta = np.abs(seen[i] - seen[i-1])
        assert np.all((delta == 0) | (delta >= np.array([0.1, 0.2]) - 1e-12))


def test_persistent_gp_warnings_use_best_measured(monkeypatch):
    install_fake_bluesky(monkeypatch)
    motor = FakeMotor(0.8)
    seen = []
    monkeypatch.setattr(gp, 'gp_kg_next_coordinate',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            gp.GPFitWarning(['ConvergenceWarning: constant_value at bound'])))
    monkeypatch.setattr(gp, '_estimate_final_minimum',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            gp.GPFitWarning(['ConvergenceWarning: length_scale at bound'])))
    result = consume(integration.minimize_plan(
        [motor], [(-1.0, 1.0)], make_acquisition(seen), simple_metrics,
        initial_coordinates=[(0.1,), (-0.5,)], initial_repeats=2,
        max_measurements=5, precisions=[3], final_position='model', verbose=False,
    ))
    best = min(result.observations, key=lambda obs: obs.metric)
    assert result.n_measurements == 5
    assert result.random_explorations == 2
    assert not result.converged
    assert not result.model_fit_ok
    assert np.isnan(result.kg)
    assert np.allclose(result.coordinate, best.coordinate)
    assert result.value == best.metric
    assert np.isclose(motor.value, best.coordinate[0])
    assert len(result.gp_warnings) == 3


def test_discrete_space_exhaustion_stops_cleanly(monkeypatch):
    install_fake_bluesky(monkeypatch)
    motor = FakeMotor(0.2)
    seen = []
    monkeypatch.setattr(gp, 'gp_kg_next_coordinate',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            gp.GPFitWarning(['ConvergenceWarning: length_scale at bound'])))
    monkeypatch.setattr(gp, '_estimate_final_minimum',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            gp.GPFitWarning(['ConvergenceWarning: length_scale at bound'])))
    result = consume(integration.minimize_plan(
        [motor], [(0, 1)], make_acquisition(seen), simple_metrics,
        initial_coordinates=[(0,), (1,)], initial_repeats=2,
        max_measurements=10, precisions=[0], verbose=False,
    ))
    assert result.n_measurements == 3
    assert result.random_explorations == 0
    assert not result.converged
    assert not result.model_fit_ok


def test_sync_search_also_recovers(monkeypatch):
    seen = []
    fit_sizes = []

    def fake_kg(observations, bounds, **kw):
        fit_sizes.append(len(observations))
        if len(observations) == 3:
            raise gp.GPFitWarning(['ConvergenceWarning: constant_value at bound'])
        return np.array([0.25]), 0.0, 0.01

    monkeypatch.setattr(gp, 'gp_kg_next_coordinate', fake_kg)
    monkeypatch.setattr(gp, '_model_minimum_or_best_measurement',
                        lambda *args, **kw: (np.array([0.2]), 0.01, True, ()))
    result = gp.search_minimum(
        accuracy=0.001,
        bounds=[(-1, 1)],
        get_measurement=lambda x: seen.append(np.array(x).copy()) or {'metric': float(x[0] ** 2)},
        extract_metrics=simple_metrics,
        initial_coordinates=[(0.1,), (-0.5,)], initial_repeats=2,
        max_measurements=8, precisions=[3], verbose=False,
    )
    assert fit_sizes == [3, 4]
    assert result.random_explorations == 1
    assert result.converged
    assert result.n_measurements == 4
    assert result.model_fit_ok


def test_final_model_warning_adds_random_point_before_convergence(monkeypatch):
    """Even if KG is clean, a warned final fit must not stop early."""
    install_fake_bluesky(monkeypatch)
    motor = FakeMotor(0.4)
    seen = []
    fit_sizes = []

    def clean_kg(observations, bounds, **kw):
        fit_sizes.append(len(observations))
        return np.array([0.3]), 0.0, 0.01  # would normally converge

    def final_fit_status(observations, *args, **kwargs):
        if len(observations) < 4:
            return observations[0].coordinate, 0.0, False, ('ConvergenceWarning: length_scale at bound',)
        return np.array([0.1]), 0.1, True, ()

    monkeypatch.setattr(gp, 'gp_kg_next_coordinate', clean_kg)
    monkeypatch.setattr(integration, '_model_minimum_or_best_measurement', final_fit_status)
    result = consume(integration.minimize_plan(
        [motor], [(-1.0, 1.0)], make_acquisition(seen), simple_metrics,
        initial_coordinates=[(0.0,), (0.5,)], initial_repeats=2,
        max_measurements=7, precisions=[3], verbose=False,
    ))
    assert fit_sizes == [3, 4]
    assert result.n_measurements == 4
    assert result.random_explorations == 1
    assert result.converged
    assert result.model_fit_ok
    assert len(result.gp_warnings) == 1
    assert all(-1 <= x[0] <= 1 for x in seen)


def test_sync_final_model_warning_prevents_premature_stopping(monkeypatch):
    fit_sizes = []

    def clean_kg(observations, bounds, **kw):
        fit_sizes.append(len(observations))
        return np.array([0.1]), 0.0, 0.01

    def final_fit_status(observations, *args, **kwargs):
        if len(observations) < 4:
            return observations[0].coordinate, 0.0, False, ('ConvergenceWarning: constant_value at bound',)
        return np.array([0.1]), 0.0, True, ()

    monkeypatch.setattr(gp, 'gp_kg_next_coordinate', clean_kg)
    monkeypatch.setattr(gp, '_model_minimum_or_best_measurement', final_fit_status)
    result = gp.search_minimum(
        accuracy=0.001, bounds=[(-1, 1)],
        get_measurement=lambda x: {'metric': float(x[0] ** 2)},
        extract_metrics=simple_metrics,
        initial_coordinates=[(0.0,), (0.5,)], initial_repeats=2,
        max_measurements=7, precisions=[3], verbose=False,
    )
    assert fit_sizes == [3, 4]
    assert result.n_measurements == 4
    assert result.random_explorations == 1
    assert result.converged
