"""Optional live, thread-safe Matplotlib monitor for N-dimensional minimisation.

Use as a *decorator of a metric callable*::

    metric = live_metric_monitor(bounds=BOUNDS, axes=(0, 1))(metric)
    metric.open()  # from the MAIN thread, before creating a Bluesky RunEngine

The metric interface is unchanged. A ``minimize_plan`` calls ``on_measurement``
for each raw record and ``on_observations`` after recomputing all metric scores.
For independent metrics the original LRU cache is unaffected; PCA metrics still
recalculate globally.

Bluesky plans execute on a worker thread (Bluesky 1.13+). The monitor performs
GP calculations on that worker thread but passes immutable plot snapshots to
the Matplotlib main-thread GUI via a lock and a Qt timer. Never manipulate
Matplotlib artists from ``on_measurement`` or ``on_observations``.

The map is the GP *posterior mean*, not the KG acquisition function. A 2-D
slice through the globally predicted N-D optimum is shown, with all measured
coordinates projected onto the displayed axes. When the GP fit issues a
warning, the model surface is hidden rather than displayed as trustworthy.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from threading import Lock, current_thread, main_thread
from typing import Any
import warnings

import numpy as np

from .gp_kg_minimizer import (
    GPFitWarning, Observation, _fit_gp, _validate_bounds,
    estimate_noise_variance,
)
from .metrics import _field
from .musr_metrics import _time_binned_asymmetry


@dataclass(frozen=True)
class PlotSnapshot:
    """Numeric data only: safe to create on a Bluesky worker thread."""

    iteration: int
    coordinates: np.ndarray
    latest_coordinate: np.ndarray
    time_us: np.ndarray
    asymmetry: np.ndarray
    asymmetry_std: np.ndarray
    metrics: np.ndarray
    surface: np.ndarray | None = None
    x_values: np.ndarray | None = None
    y_values: np.ndarray | None = None
    model_minimum: np.ndarray | None = None
    model_value: float | None = None
    fit_status: str = "Waiting for initial-design metrics"


def _validate_axes(axes: tuple[int, int] | None, n_dimensions: int) -> tuple[int, int]:
    if n_dimensions < 2:
        raise ValueError("The 2D monitor requires at least two optimisation axes")
    if axes is None:
        return (0, 1)
    if (
        not isinstance(axes, (tuple, list)) or len(axes) != 2
        or any(isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer)) for a in axes)
        or axes[0] == axes[1]
        or any(a < 0 or a >= n_dimensions for a in axes)
    ):
        raise ValueError(f"axes must be two different indices from 0 to {n_dimensions-1}")
    return (int(axes[0]), int(axes[1]))


def _latest_asymmetry(record: Any, *, alpha: float,
                      rebin: int, time_window_ns: tuple[float, float] | None,
                      min_bin_counts: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build plot arrays from the same live-DAE/synthetic counts as the metric."""
    if time_window_ns is None:
        t = np.asarray(_field(record, "time_bins"), dtype=float).reshape(-1)
        if len(t) == 0:
            raise ValueError("time_bins is empty")
        # Use bin centres, and include the final bin by extending the bound.
        upper = np.nextafter(float(t[-1]), np.inf)
        time_window_ns = (float(t[0]), upper)
    t, a, sigma, counts = _time_binned_asymmetry(
        record, alpha=alpha, rebin=rebin, time_window_ns=time_window_ns,
    )
    valid = (
        np.isfinite(a) & np.isfinite(sigma) & (sigma > 0)
        & (counts >= min_bin_counts)
    )
    return t / 1000.0, np.where(valid, a, np.nan), np.where(valid, sigma, np.nan)


def gp_mean_slice(
    observations: Sequence[Observation],
    bounds: Sequence[Sequence[float]],
    *, axes: tuple[int, int] | None = None,
    grid_size: int = 65,
    minimum_points: int = 1024,
    random_state: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a trustworthy GP once and return its N-D optimum and 2D mean cut.

    The global minimum is searched on a deterministic Sobol point set plus all
    observed positions. On an N>2 scan, all non-displayed dimensions are held
    at that global posterior-mean minimum. The coloured surface is computed
    using the *same fitted GP*. ``GPFitWarning`` is passed to the caller and
    the warned model must not be rendered.

    Returns
    -------
    (x_grid, y_grid, metric_grid, estimated_global_minimum, minimum_metric)
    """
    from scipy.stats import qmc

    b = _validate_bounds(bounds)
    axis_x, axis_y = _validate_axes(axes, len(b))
    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)) or not 8 <= grid_size <= 300:
        raise ValueError("grid_size must be an integer from 8 to 300")
    if isinstance(minimum_points, (bool, np.bool_)) or not isinstance(minimum_points, (int, np.integer)) or not 2 <= minimum_points <= 65536:
        raise ValueError("minimum_points must be an integer from 2 to 65536")
    if not observations:
        raise ValueError("Cannot plot a GP without observations")
    noise_variance = estimate_noise_variance(observations)
    gp, offset, scale, _ = _fit_gp(
        observations, b, noise_variance, random_state=int(random_state),
    )

    # Search the FULL N-dimensional space first. Include measured positions.
    n_sobol = int(minimum_points)
    sampler = qmc.Sobol(d=len(b), scramble=True, seed=int(random_state))
    unit_points = sampler.random_base2(m=int(np.ceil(np.log2(n_sobol))))[:n_sobol]
    measured_unit = np.stack([
        (np.asarray(obs.coordinate, dtype=float) - b[:, 0]) / (b[:, 1] - b[:, 0])
        for obs in observations
    ])
    unit_points = np.concatenate((unit_points, measured_unit), axis=0)
    predicted = gp.predict(unit_points) * scale + offset
    index = int(np.argmin(predicted))
    estimated_minimum = b[:, 0] + unit_points[index] * (b[:, 1] - b[:, 0])
    minimum_metric = float(predicted[index])

    xv = np.linspace(b[axis_x, 0], b[axis_x, 1], grid_size)
    yv = np.linspace(b[axis_y, 0], b[axis_y, 1], grid_size)
    xx, yy = np.meshgrid(xv, yv, indexing="xy")
    cut_points = np.tile(estimated_minimum, (xx.size, 1))
    cut_points[:, axis_x] = xx.ravel()
    cut_points[:, axis_y] = yy.ravel()
    cut_unit = (cut_points - b[:, 0]) / (b[:, 1] - b[:, 0])
    values = gp.predict(cut_unit) * scale + offset
    return xv, yv, values.reshape(xx.shape), estimated_minimum, minimum_metric


class LiveOptimizationMonitor:
    """Manage one Matplotlib window without touching Qt from plan threads."""

    def __init__(self, bounds, *, axes=None, axis_labels=None,
                 alpha: float = 1.0, asymmetry_rebin: int = 8,
                 asymmetry_time_window_ns: tuple[float, float] | None = None,
                 min_bin_counts: float = 10.0,
                 grid_size: int = 65, minimum_points: int = 1024,
                 random_state: int = 12345,
                 show: bool = True, save_path: str | None = None,
                 refresh_interval_ms: int = 150):
        self.bounds = _validate_bounds(bounds)
        self.axes = _validate_axes(axes, len(self.bounds))
        if axis_labels is not None and len(axis_labels) != len(self.bounds):
            raise ValueError("axis_labels must have one label per optimisation dimension")
        self.axis_labels = tuple(str(v) for v in axis_labels) if axis_labels is not None else None
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not isinstance(asymmetry_rebin, int) or isinstance(asymmetry_rebin, bool) or asymmetry_rebin < 1:
            raise ValueError("asymmetry_rebin must be a positive integer")
        if not np.isfinite(min_bin_counts) or min_bin_counts <= 0:
            raise ValueError("min_bin_counts must be finite and positive")
        if (isinstance(grid_size, bool) or not isinstance(grid_size, int) or not 8 <= grid_size <= 300):
            raise ValueError("grid_size must be an integer between 8 and 300")
        if (isinstance(minimum_points, bool) or not isinstance(minimum_points, int) or not 2 <= minimum_points <= 65536):
            raise ValueError("minimum_points must be an integer between 2 and 65536")
        if isinstance(refresh_interval_ms, bool) or not isinstance(refresh_interval_ms, int) or refresh_interval_ms < 20:
            raise ValueError("refresh_interval_ms must be an integer >= 20")
        if not show and not save_path:
            raise ValueError("At least one of show=True or save_path must be provided")
        self.alpha = float(alpha)
        self.asymmetry_rebin = asymmetry_rebin
        self.asymmetry_time_window_ns = asymmetry_time_window_ns
        self.min_bin_counts = min_bin_counts
        self.grid_size = grid_size
        self.minimum_points = minimum_points
        self.random_state = int(random_state)
        self.show = bool(show)
        self.save_path = save_path
        self.refresh_interval_ms = refresh_interval_ms

        self._lock = Lock()
        self._render_lock = Lock()  # headless PNG rendering may run on worker
        self._pending: PlotSnapshot | None = None
        self._last: PlotSnapshot | None = None
        self._disabled = False
        self._warning_printed = False
        self._active = False
        self.figure = None
        self.ax_map = None
        self.ax_asym = None
        self._timer = None
        self._map_artist = None
        self._points_artist = None
        self._minimum_artist = None
        self._colorbar = None

    def set_axis_labels(self, labels: Sequence[str]) -> None:
        if self.axis_labels is None:
            if len(labels) != len(self.bounds):
                raise ValueError("Incorrect number of motor names")
            self.axis_labels = tuple(str(v) for v in labels)

    def open(self) -> "LiveOptimizationMonitor":
        """Create Qt artists on the main thread, BEFORE the Bluesky RunEngine.

        With a headless Matplotlib backend use show=False, save_path='...png'
        instead. A Qt backend is needed for a *live GUI* on Windows. Fail
        before motor movement if GUI prerequisites are unavailable.
        """
        if self.figure is not None:
            return self
        if current_thread() is not main_thread():
            raise RuntimeError("Call metric.open() from the main thread before RunEngine creation")
        import matplotlib
        if self.show and "qt" not in matplotlib.get_backend().lower():
            raise RuntimeError(
                f"Live Bluesky plot requires the QtAgg Matplotlib backend; "
                f"found {matplotlib.get_backend()!r}. Install a Qt binding "
                "(PyQt5/PyQt6/PySide6) and set MPLBACKEND=qtagg before launching. "
                "Alternatively use show=False with save_path to save PNG snapshots."
            )
        import matplotlib.pyplot as plt
        fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.1), layout="constrained")
        try:
            fig.canvas.manager.set_window_title("N-D optimisation: model and latest muon asymmetry")
        except AttributeError:
            pass
        self.figure, self.ax_map, self.ax_asym = fig, left, right
        self._map_artist = left.imshow(
            np.zeros((2, 2)), origin="lower", aspect="auto", interpolation="nearest",
            extent=(self.bounds[self.axes[0], 0], self.bounds[self.axes[0], 1],
                    self.bounds[self.axes[1], 0], self.bounds[self.axes[1], 1]),
            cmap="viridis", visible=False,
        )
        self._colorbar = fig.colorbar(self._map_artist, ax=left, label="GP posterior mean (metric)")
        self._colorbar.ax.set_visible(False)
        self._points_artist = left.scatter(
            [], [], s=46, facecolors="none", edgecolors="black",
            linewidths=1.35, zorder=5, label="Measured points (projection)",
        )
        self._minimum_artist = left.scatter(
            [], [], s=145, marker="*", facecolors="white", edgecolors="black",
            linewidths=1.25, zorder=6, label="GP estimated minimum",
        )
        left.set_xlim(*self.bounds[self.axes[0]])
        left.set_ylim(*self.bounds[self.axes[1]])
        left.set_xlabel(self._label(self.axes[0]))
        left.set_ylabel(self._label(self.axes[1]))
        left.legend(loc="upper right", fontsize=8)
        left.set_title("Waiting for measurements")
        right.set_xlabel("Time (µs)")
        right.set_ylabel("Muon asymmetry")
        right.set_title("Latest acquisition")
        right.grid(alpha=0.25)
        if self.show:
            plt.show(block=False)
            self._timer = fig.canvas.new_timer(interval=self.refresh_interval_ms)
            self._timer.add_callback(self._timer_callback)
            self._timer.start()
            fig.canvas.mpl_connect("close_event", lambda event: self.close())
        return self

    def _timer_callback(self) -> None:
        self._sync_visibility()
        self.render_pending()

    def activate(self) -> None:
        """Show a prepared GUI monitor when its optimisation stage starts."""
        self._active = True

    def deactivate(self) -> None:
        """Hide a GUI monitor when its optimisation stage finishes."""
        self._active = False

    def _sync_visibility(self) -> None:
        if self.figure is None or not self.show:
            return
        window = getattr(self.figure.canvas.manager, "window", None)
        if window is not None:
            if self._active:
                window.show()
                self.figure.canvas.draw_idle()
            else:
                window.hide()

    def _label(self, index: int) -> str:
        return self.axis_labels[index] if self.axis_labels is not None else f"Axis {index + 1}"

    def _publish(self, snapshot: PlotSnapshot) -> None:
        with self._lock:
            self._pending = snapshot
        if not self.show and self.figure is not None:
            # In headless PNG mode there is no Qt event loop, and the dedicated
            # Agg figure can be rendered synchronously by the plan worker.
            self.render_pending()

    def _build_snapshot(self, observations: Sequence[Observation]) -> PlotSnapshot:
        latest = observations[-1]
        time, asymmetry, sigma = _latest_asymmetry(
            latest.measurement, alpha=self.alpha,
            rebin=self.asymmetry_rebin,
            time_window_ns=self.asymmetry_time_window_ns,
            min_bin_counts=self.min_bin_counts,
        )
        return PlotSnapshot(
            iteration=len(observations),
            coordinates=np.stack([np.asarray(o.coordinate, float).copy() for o in observations]),
            latest_coordinate=np.asarray(latest.coordinate, float).copy(),
            time_us=time.copy(), asymmetry=asymmetry.copy(),
            asymmetry_std=sigma.copy(),
            metrics=np.asarray([o.metric for o in observations], dtype=float),
            fit_status="Waiting for model update",
        )

    def on_measurement(self, observations: Sequence[Observation]) -> None:
        """Immediately publish latest raw histogram, including initial scans."""
        if self._disabled or not observations:
            return
        try:
            self._publish(self._build_snapshot(observations))
        except Exception as exc:
            self._warn_once(exc)

    def on_observations(self, observations: Sequence[Observation],
                        estimated_noise_std: float | None) -> None:
        """Update the 2D GP cut and latest asymmetry *after* all metrics update.

        This method does no Matplotlib operations: it is safe to call from
        Bluesky's RunEngine worker thread. It always publishes a new snapshot,
        even when the GP is not ready or warns.
        """
        if self._disabled or not observations:
            return
        # Rebuild from the observations instead of reading _pending: the Qt
        # timer may consume _pending concurrently. Otherwise an update could
        # occasionally lose its GP surface after publishing the raw histogram.
        try:
            snapshot = self._build_snapshot(observations)
        except Exception as exc:
            self._warn_once(exc)
            return
        if estimated_noise_std is None:
            self._publish(replace(snapshot, fit_status="Waiting for repeated-point noise estimate"))
            return
        try:
            xv, yv, mean, x_min, y_min = gp_mean_slice(
                observations, self.bounds, axes=self.axes,
                grid_size=self.grid_size, minimum_points=self.minimum_points,
                random_state=self.random_state + len(observations),
            )
            status = "GP posterior mean"
            if len(self.bounds) > 2:
                axes_hidden = [i for i in range(len(self.bounds)) if i not in self.axes]
                fixed = ", ".join(f"{self._label(i)}={x_min[i]:.4g}" for i in axes_hidden)
                status += f" | cut at {fixed}"
            self._publish(replace(
                snapshot, surface=mean, x_values=xv, y_values=yv,
                model_minimum=x_min, model_value=y_min,
                fit_status=status,
            ))
        except GPFitWarning:
            self._publish(replace(snapshot, fit_status="GP warning: model surface withheld"))
        except Exception as exc:
            # Plotting cannot abort a live experimental run. Existing failures
            # from the optimizer itself remain unmodified.
            self._warn_once(exc)
            self._publish(replace(snapshot, fit_status=f"Plot unavailable: {type(exc).__name__}"))

    def _warn_once(self, exc: Exception) -> None:
        if not self._warning_printed:
            warnings.warn(f"Live monitor unavailable: {type(exc).__name__}: {exc}",
                          RuntimeWarning, stacklevel=2)
            self._warning_printed = True

    def render_pending(self) -> None:
        """Redraw pending plot data on the GUI main thread or in a headless test."""
        if self.figure is None:
            return
        if self.show and current_thread() is not main_thread():
            raise RuntimeError("Qt Matplotlib artists must be drawn on the main thread")
        with self._render_lock:
            self._render_snapshot()

    def _render_snapshot(self) -> None:
        with self._lock:
            snapshot = self._pending
            self._pending = None
        if snapshot is None:
            return
        self._last = snapshot
        left, right = self.ax_map, self.ax_asym
        ix, iy = self.axes
        self._points_artist.set_offsets(snapshot.coordinates[:, (ix, iy)])
        if snapshot.surface is not None:
            self._map_artist.set_data(snapshot.surface)
            self._map_artist.set_extent((
                snapshot.x_values[0], snapshot.x_values[-1],
                snapshot.y_values[0], snapshot.y_values[-1],
            ))
            finite = snapshot.surface[np.isfinite(snapshot.surface)]
            if finite.size:
                vmin, vmax = float(np.min(finite)), float(np.max(finite))
                if vmax <= vmin:
                    pad = max(1e-10, abs(vmin) * 1e-6)
                    vmin -= pad
                    vmax += pad
                self._map_artist.set_clim(vmin, vmax)
                self._map_artist.set_visible(True)
                self._colorbar.ax.set_visible(True)
                self._colorbar.update_normal(self._map_artist)
            if snapshot.model_minimum is not None:
                self._minimum_artist.set_offsets(snapshot.model_minimum[[ix, iy]][None, :])
            else:
                self._minimum_artist.set_offsets(np.empty((0, 2)))
        else:
            self._map_artist.set_visible(False)
            self._colorbar.ax.set_visible(False)
            self._minimum_artist.set_offsets(np.empty((0, 2)))
        left.set_xlabel(self._label(ix))
        left.set_ylabel(self._label(iy))
        left.set_title(f"N={snapshot.iteration} | {snapshot.fit_status}", fontsize=10)
        right.clear()
        time = snapshot.time_us
        asymmetry = snapshot.asymmetry
        std = snapshot.asymmetry_std
        right.plot(time, asymmetry, lw=1.2, label="Measured asymmetry")
        right.fill_between(time, asymmetry - std, asymmetry + std,
                           alpha=0.18, label="Poisson ±1σ")
        right.set_xlabel("Time (µs)")
        right.set_ylabel("Muon asymmetry")
        right.grid(alpha=0.25)
        right.legend(loc="upper right", fontsize=8)
        coord = np.array2string(snapshot.latest_coordinate, precision=3)
        metric = snapshot.metrics[-1]
        suffix = f" | metric={metric:.5g}" if np.isfinite(metric) else ""
        right.set_title(f"Latest measurement #{snapshot.iteration}, x={coord}{suffix}", fontsize=10)
        self.figure.canvas.draw_idle()
        if self.save_path:
            self.figure.savefig(self.save_path, dpi=125)

    def close(self) -> None:
        """Stop redraws and close the monitor figure."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self.figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.figure)
            self.figure = None
            self.ax_map = None
            self.ax_asym = None
            self._map_artist = None
            self._points_artist = None
            self._minimum_artist = None
            self._colorbar = None


class MonitoredMetric:
    """Transparent metric-callable wrapper with optional progress hooks."""
    def __init__(self, metric: Callable, monitor: LiveOptimizationMonitor):
        if not callable(metric):
            raise TypeError("The decorated metric must be callable")
        self.metric = metric
        self.monitor = monitor

    def __call__(self, measurements, first_set=False):
        return self.metric(measurements, first_set)

    @property
    def changes_with_dataset(self) -> bool:
        return bool(getattr(self.metric, "changes_with_dataset", True))

    def __getattr__(self, name):
        # Retain IndependentMetric.cache_info/cache_clear and other methods.
        return getattr(self.metric, name)

    def open(self):
        return self.monitor.open()

    def close(self):
        return self.monitor.close()

    def activate(self):
        return self.monitor.activate()

    def deactivate(self):
        return self.monitor.deactivate()

    def set_axis_labels(self, labels):
        self.monitor.set_axis_labels(labels)

    def on_measurement(self, observations):
        self.monitor.on_measurement(observations)

    def on_observations(self, observations, estimated_noise_std):
        self.monitor.on_observations(observations, estimated_noise_std)


def live_metric_monitor(*, bounds, axes=None, axis_labels=None,
                        alpha: float = 1.0, asymmetry_rebin: int = 8,
                        asymmetry_time_window_ns=None,
                        min_bin_counts: float = 10.0, grid_size: int = 65,
                        minimum_points: int = 1024, random_state: int = 12345,
                        show: bool = True, save_path: str | None = None):
    """Decorator factory for any metric with a live 2D GP/asymmetry display.

    Works for both independent/cached metrics and global/PCA metrics::

        metric = live_metric_monitor(bounds=bounds, axes=None)(
            integrated_asymmetry_metric(...)
        )
        metric.open()  # main thread, BEFORE creating the Bluesky RunEngine

    ``axes=(i,j)`` selects the two scan dimensions; default is ``(0,1)``.
    Other dimensions are fixed at the GP-predicted *global* N-D optimum.
    Measurements are displayed as black circles, projected onto those axes.
    The map is the predicted metric (lower means better). If fitting warns,
    the model map is withheld; the latest time-resolved asymmetry continues.
    """
    def decorate(metric):
        return MonitoredMetric(metric, LiveOptimizationMonitor(
            bounds, axes=axes, axis_labels=axis_labels,
            alpha=alpha, asymmetry_rebin=asymmetry_rebin,
            asymmetry_time_window_ns=asymmetry_time_window_ns,
            min_bin_counts=min_bin_counts, grid_size=grid_size,
            minimum_points=minimum_points, random_state=random_state,
            show=show, save_path=save_path,
        ))
    return decorate
