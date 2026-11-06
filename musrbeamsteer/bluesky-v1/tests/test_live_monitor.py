"""Live Matplotlib monitor: no EPICS/RunEngine/Qt required in these tests."""
from __future__ import annotations

import threading

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..blueskysteercore import live_monitor as monitor
from ..blueskysteercore.gp_kg_minimizer import GPFitWarning, Observation
from ..blueskysteercore.musr_metrics import (
    integrated_asymmetry_metric, pca_asymmetry_metric,
)


def make_record(uid, *, scale=1.0):
    t = np.arange(64) * 16 + 8
    f = np.asarray(1000 * scale * np.exp(-t / 5000) * (1 + 0.12*np.exp(-t/700)), float)
    b = np.asarray(1000 * scale * np.exp(-t / 5000) * (1 - 0.12*np.exp(-t/700)), float)
    return {"uid": uid, "data": {
        "time_bins": t,
        "forward_counts": f,
        "backward_counts": b,
    }}


def observations(*coords):
    return [Observation(coordinate=np.asarray(x, float),
                        measurement=make_record(str(i), scale=1 + 0.1 * i),
                        metric=float(i) * .02)
            for i, x in enumerate(coords)]


def test_2d_axes_validation_and_default():
    b = ((-1, 1), (-2, 2), (3, 5))
    assert monitor._validate_axes(None, 3) == (0, 1)
    assert monitor._validate_axes((2, 0), 3) == (2, 0)
    for bad in [(0, 0), (3, 1), (0, -1), (True, 1), (0,), (0, 1, 2)]:
        with pytest.raises(ValueError, match="axes"):
            monitor.LiveOptimizationMonitor(b, axes=bad, show=False, save_path="demo.png")
    with pytest.raises(ValueError, match="at least two"):
        monitor.LiveOptimizationMonitor([(-1, 1)], show=False, save_path="demo.png")


def test_gp_mean_cut_holds_hidden_dimensions_at_global_optimum(monkeypatch):
    b = np.asarray(((0, 10), (-2, 2), (100, 200)), float)
    target = np.asarray([.25, .62, .8])
    calls = []

    class FakeGP:
        def predict(self, unit_coords):
            calls.append(np.asarray(unit_coords).copy())
            return np.sum((unit_coords - target) ** 2, axis=1)

    monkeypatch.setattr(monitor, "_fit_gp", lambda *a, **kw: (FakeGP(), 0.0, 1.0, .01))
    obs = observations((1, 0, 130), (1, 0, 130), (7, -1, 170))
    x, y, grid, best, predicted = monitor.gp_mean_slice(
        obs, b, axes=(0, 2), grid_size=15, minimum_points=512,
    )
    assert grid.shape == (15, 15)
    assert x[[0, -1]] == pytest.approx([0, 10])
    assert y[[0, -1]] == pytest.approx([100, 200])
    assert best == pytest.approx([2.5, .48, 180], abs=3.5)
    assert predicted >= 0
    # Last GP.predict call is the 2D slice; hidden axis 1 is fixed at global best.
    cut = calls[-1]
    assert cut.shape == (225, 3)
    assert np.allclose(cut[:, 1], (best[1] - b[1, 0]) / (b[1, 1] - b[1, 0]))
    assert np.unique(cut[:, 0]).size == 15
    assert np.unique(cut[:, 2]).size == 15


def test_decorator_is_transparent_for_independent_cache_and_global_pca(tmp_path):
    decorator = monitor.live_metric_monitor(
        bounds=[(-1, 1), (-2, 2)], show=False,
        save_path=str(tmp_path / "snapshot.png"),
    )
    independent = decorator(integrated_asymmetry_metric(time_window_ns=(0, 1000), rebin=2))
    records = [make_record("a"), make_record("b", scale=1.1)]
    first = independent(records)
    assert independent(records) == first
    assert independent.cache_info().hits == 2
    assert not independent.changes_with_dataset
    independent.cache_clear()
    assert independent.cache_info().currsize == 0
    global_metric = decorator(pca_asymmetry_metric(time_window_ns=(0, 1000), rebin=2))
    assert global_metric.changes_with_dataset
    assert len(global_metric(records)) == 2


def test_headless_monitor_writes_png_and_latest_asymmetry(tmp_path, monkeypatch):
    path = tmp_path / "latest.png"
    m = monitor.LiveOptimizationMonitor(
        bounds=[(-.5, .5), (-1.5, 1.5)], show=False,
        save_path=str(path), alpha=1, asymmetry_rebin=4,
        grid_size=8, minimum_points=32,
    )
    m.open()
    obs = observations((0.12, -.3), (0.12, -.3))
    m.on_measurement(obs)
    m.render_pending()
    assert path.exists() and path.stat().st_size > 10_000
    assert m._last.iteration == 2
    assert m._last.time_us.size == 16
    assert m.ax_asym.lines
    assert m.ax_map.get_xlabel() == "Axis 1"
    assert m._map_artist.get_visible() is False

    xv = np.linspace(-.5, .5, 8)
    yv = np.linspace(-1.5, 1.5, 8)
    surface = np.arange(64, dtype=float).reshape((8, 8))
    monkeypatch.setattr(monitor, "gp_mean_slice", lambda *a, **kw: (
        xv, yv, surface, np.asarray([.1, -.3]), 1.23,
    ))
    m.on_observations(obs, .01)
    m.render_pending()
    assert m._map_artist.get_visible()
    assert m._colorbar.ax.get_visible()
    assert len(m._points_artist.get_offsets()) == 2
    assert m._last.model_value == 1.23
    assert path.stat().st_size > 10_000
    m.set_axis_labels(["KEPCO_04 (A)", "KEPCO_05 (A)"])
    m.on_measurement(obs)
    m.render_pending()
    assert m.ax_map.get_xlabel() == "KEPCO_04 (A)"
    plt.close(m.figure)


def test_warned_fit_hides_old_surface_but_preserves_asymmetry(monkeypatch, tmp_path):
    m = monitor.LiveOptimizationMonitor(
        bounds=[(-1, 1), (-1, 1)], show=False,
        save_path=str(tmp_path / "monitor.png"),
    )
    m.open()
    obs = observations((0., 0.), (0., 0.), (.5, .5))
    monkeypatch.setattr(monitor, "gp_mean_slice", lambda *a, **kw: (
        np.linspace(-1, 1, 8), np.linspace(-1, 1, 8), np.ones((8, 8)),
        np.zeros(2), 1.0,
    ))
    m.on_observations(obs, .1)
    m.render_pending()
    assert m._map_artist.get_visible()

    def warned(*args, **kwargs):
        raise GPFitWarning(("ConvergenceWarning: length_scale",))
    monkeypatch.setattr(monitor, "gp_mean_slice", warned)
    m.on_observations(obs, .1)
    m.render_pending()
    assert not m._map_artist.get_visible()
    assert not m._colorbar.ax.get_visible()
    assert "warning" in m._last.fit_status
    assert np.isfinite(m._last.asymmetry).any()
    plt.close(m.figure)


def test_headless_png_can_be_updated_from_worker(tmp_path):
    m = monitor.LiveOptimizationMonitor(
        bounds=[(-1, 1), (-1, 1)], show=False,
        save_path=str(tmp_path / "monitor.png"),
    )
    m.open()
    def update_from_worker():
        # Headless PNG mode can render its private Agg figure from worker.
        # Qt GUI mode instead redraws from its main-thread timer.
        m.on_measurement(observations((0., 0.)))

    thread = threading.Thread(target=update_from_worker)
    thread.start()
    thread.join()
    assert m._last.iteration == 1
    assert (tmp_path / "monitor.png").exists()
    plt.close(m.figure)


def test_monitor_validates_before_any_hardware_access(monkeypatch):
    with pytest.raises(RuntimeError, match="QtAgg"):
        monitor.LiveOptimizationMonitor(
            bounds=[(-1, 1), (-1, 1)], show=True,
        ).open()


def test_bluesky_plan_calls_monitor_hooks_on_every_measurement(monkeypatch):
    """Even initial-design measurements notify the metric, before first PCA fit."""
    import sys
    import types
    from musrbeamsteer import bluesky_integration as integ
    from musrbeamsteer.gp_kg_minimizer import NextCoordinate

    class Motor:
        def __init__(self, name):
            self.name = name
            self.value = 0.

    def rd(m):
        if False:
            yield
        return m.value

    def mv(*args):
        for m, value in zip(args[::2], args[1::2]):
            m.value = float(value)
        if False:
            yield

    def sleep(seconds):
        if False:
            yield

    fake = types.ModuleType('bluesky')
    fake.__path__ = []
    stubs = types.SimpleNamespace(rd=rd, mv=mv, sleep=sleep)
    fake.plan_stubs = stubs
    monkeypatch.setitem(sys.modules, 'bluesky', fake)
    monkeypatch.setitem(sys.modules, 'bluesky.plan_stubs', stubs)

    monitor_calls = []
    model_calls = []
    labels = []

    class TrackingMetric:
        def __call__(self, records, first_set=False):
            return [r['metric'] for r in records]
        def on_measurement(self, obs):
            monitor_calls.append(len(obs))
        def on_observations(self, obs, noise):
            model_calls.append((len(obs), noise))
        def set_axis_labels(self, values):
            labels.extend(values)

    def acq(x):
        if False:
            yield
        return {"metric": float((x[0] - .1) ** 2 + (x[1] + .3) ** 2 + .001)}

    monkeypatch.setattr(integ, 'next_coordinate_with_warning_recovery',
                        lambda *a, **k: NextCoordinate(np.array([.2, -.2]), 1., .01))
    monkeypatch.setattr(integ, '_model_minimum_or_best_measurement',
                        lambda *a, **k: (np.array([.1, -.3]), .01, True, ()))

    def consume(gen):
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            return exc.value

    result = consume(integ.minimize_plan(
        [Motor('one'), Motor('two')],
        [(-1, 1), (-1, 1)], acq, TrackingMetric(),
        initial_coordinates=[(0., 0.), (.5, .5)],
        max_measurements=4, verbose=0,
    ))
    assert labels == ['one', 'two']
    assert monitor_calls == [1, 2, 3, 4]
    assert [n for n, noise in model_calls] == [3, 4]
    assert result.n_measurements == 4
