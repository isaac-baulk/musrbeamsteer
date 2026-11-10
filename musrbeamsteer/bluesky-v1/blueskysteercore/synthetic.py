"""Beam-off MUSR-like histograms for N-dimensional Bluesky experiments.

The spatial Kubo--Toyabe mixing fraction and the total count rate are separate:

* ``w(x) = exp(-sum(curvature_i * (x_i-center_i)**2))`` controls the
  *shape* of the time-dependent polarisation. A flat-polarisation background
  has the same t=0 asymmetry as the static KT sample.
* ``R(x)/R(center) = edge_rate_fraction ** r(x)**2`` controls the *total*
  muons per good frame. ``r`` is the largest fraction of the distance to a
  boundary in any dimension, measured along the chosen side from ``center``.
  Consequently r=0 at the centre and r=1 on EVERY face of the rectangular
  parameter space, including its corners. This is a convenient test profile,
  not a physical model of beam transport.

The histogram contains Poisson-distributed counts in each detector and time
bin. Its expectation decays as exp(-t/tau_mu), with per-bin sigma=sqrt(E[N]).
Default bins have WIDTH 16 ns, with edges from 0 through 32000 ns (2000 bins).
A shared numpy.random.Generator gives reproducible *new* noise at every call.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .isis_dae import AcquisitionDuration, DurationUnit, DetectorGrouping, musr_longitudinal_grouping


DEFAULT_BIN_WIDTH_NS = 16.0
DEFAULT_MAX_TIME_US = 32.0
DEFAULT_N_TIME_BINS = 2000


@dataclass(frozen=True)
class SyntheticMotor:
    """Configuration for a soft Bluesky motor with setpoint and readback PVs.

    The PV names are metadata only: ``build()`` creates in-memory signals and
    never connects to EPICS. ``safe_bounds`` are enforced by the device before
    a setpoint is accepted.
    """

    name: str
    setpoint_pv: str
    readback_pv: str
    safe_bounds: tuple[float, float]
    initial_value: float = 0.0
    readback_tolerance: float = 0.01
    timeout: float = 30.0

    def __post_init__(self) -> None:
        lower, upper = self.safe_bounds
        if not self.name:
            raise ValueError("name is required")
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("safe_bounds must be finite increasing bounds")
        if not lower <= self.initial_value <= upper:
            raise ValueError("initial_value must be within safe_bounds")
        if not math.isfinite(self.readback_tolerance) or self.readback_tolerance <= 0:
            raise ValueError("readback_tolerance must be finite and > 0")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and > 0")

    def build(self):
        """Build an in-memory Ophyd Async device for this motor configuration."""
        from ophyd_async.core import (
            AsyncStatus,
            HintedSignal,
            StandardReadable,
            soft_signal_r_and_setter,
            wait_for_value,
        )

        configuration = self

        class SoftMotor(StandardReadable):
            def __init__(self):
                with self.add_children_as_readables(HintedSignal):
                    self.readback, self._set_readback = soft_signal_r_and_setter(
                        float, configuration.initial_value
                    )
                self.setpoint, self._set_setpoint = soft_signal_r_and_setter(
                    float, configuration.initial_value
                )
                super().__init__(name=configuration.name)

            @AsyncStatus.wrap
            async def set(self, value: float) -> None:
                target = float(value)
                lower, upper = configuration.safe_bounds
                if not math.isfinite(target) or not lower <= target <= upper:
                    raise ValueError(
                        f"{configuration.name} target {target} outside "
                        f"safe_bounds {configuration.safe_bounds}"
                    )
                self._set_setpoint(target)
                self._set_readback(target)
                await wait_for_value(
                    self.readback,
                    lambda actual: abs(float(actual) - target)
                    <= configuration.readback_tolerance,
                    timeout=configuration.timeout,
                )

        device = SoftMotor()
        device.setpoint_pv = configuration.setpoint_pv
        device.readback_pv = configuration.readback_pv
        device.safe_bounds = configuration.safe_bounds
        return device


def static_gaussian_kubo_toyabe(time_us: Sequence[float] | np.ndarray, delta_us_inv: float) -> np.ndarray:
    """Static Gaussian Kubo--Toyabe with t in microseconds, delta in us^-1."""
    delta = float(delta_us_inv)
    times = np.asarray(time_us, dtype=float)
    if not np.isfinite(delta) or delta < 0 or not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("delta_us_inv and time_us must be finite and nonnegative")
    squared = np.square(delta * times)
    return 1 / 3 + (2 / 3) * (1 - squared) * np.exp(-squared / 2)


def spatial_rate_fraction(
    coordinate: Sequence[float] | np.ndarray,
    *,
    center: Sequence[float],
    bounds: Sequence[Sequence[float]],
    edge_rate_fraction: float = 0.10,
) -> float:
    """Relative muons/frame; 1 at centre, exactly edge_rate_fraction on box faces.

    Each axis is scaled by the centre-to-upper/lower-face distance depending
    on the direction of travel. The Chebyshev radius (maximum scaled axis
    distance) means any point on *any* boundary, including corners, has r=1.
    All dimensions are supported and no detector, magnet or unit is assumed.
    """
    x = np.asarray(coordinate, dtype=float)
    c = np.asarray(center, dtype=float)
    b = np.asarray(bounds, dtype=float)
    if x.ndim != 1 or x.size == 0 or c.shape != x.shape or b.shape != (x.size, 2):
        raise ValueError("coordinate, center, bounds must have compatible nonzero N-D shapes")
    if not (np.isfinite(x).all() and np.isfinite(c).all() and np.isfinite(b).all()):
        raise ValueError("coordinate, center and bounds must be finite")
    if np.any(b[:, 0] >= b[:, 1]):
        raise ValueError("Every bound must satisfy lower < upper")
    # A nontrivial drop at every face is incompatible with a centre on a face.
    if np.any(c <= b[:, 0]) or np.any(c >= b[:, 1]):
        raise ValueError("center must be strictly inside bounds")
    if np.any(x < b[:, 0]) or np.any(x > b[:, 1]):
        raise ValueError("coordinate lies outside bounds")
    if not np.isfinite(edge_rate_fraction) or not 0 < edge_rate_fraction <= 1:
        raise ValueError("edge_rate_fraction must be in (0, 1]")

    distance_to_face = np.where(x >= c, b[:, 1] - c, c - b[:, 0])
    radius = float(np.max(np.abs((x - c) / distance_to_face)))
    return float(edge_rate_fraction ** (radius * radius))


def synthetic_good_frames_and_seconds(
    duration: AcquisitionDuration,
    *,
    counts_per_frame: float,
    good_frames_per_second: float,
) -> tuple[int, float]:
    """Simulated exposure. Events use the EXPECTED position-dependent rate.

    On an event target, edge positions need more good frames than the centre.
    Actual Poisson totals fluctuate about the target, unlike a hardware
    event-threshold waiter. The other modes fix exposure independent of rate.
    """
    if counts_per_frame <= 0 or not np.isfinite(counts_per_frame):
        raise ValueError("counts_per_frame must be finite and > 0")
    if good_frames_per_second <= 0 or not np.isfinite(good_frames_per_second):
        raise ValueError("good_frames_per_second must be finite and > 0")
    if duration.unit == DurationUnit.GOOD_FRAMES:
        frames = int(duration.value)
        seconds = frames / good_frames_per_second
    elif duration.unit == DurationUnit.EVENTS:
        frames = max(1, int(np.ceil(float(duration.value) / counts_per_frame)))
        seconds = frames / good_frames_per_second
    elif duration.unit == DurationUnit.SECONDS:
        seconds = float(duration.value)
        frames = max(1, int(np.rint(seconds * good_frames_per_second)))
    else:
        raise ValueError(f"Unsupported duration unit: {duration.unit!r}")
    return frames, float(seconds)


def generate_synthetic_musr_data(
    coordinate: Sequence[float] | np.ndarray,
    *,
    center: Sequence[float],
    curvature: Sequence[float],
    bounds: Sequence[Sequence[float]],
    delta_us_inv: float,
    initial_asymmetry: float = 0.23,
    duration: AcquisitionDuration = AcquisitionDuration.good_frames(500),
    counts_per_frame: float = 1000.0,
    edge_rate_fraction: float = 0.10,
    good_frames_per_second: float = 50.0,
    muon_lifetime_us: float = 2.1969811,
    n_time_bins: int = DEFAULT_N_TIME_BINS,
    max_time_us: float = DEFAULT_MAX_TIME_US,
    grouping: DetectorGrouping | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate one DAE-shaped measurement, without reading or writing a NeXus file.

    Example::

        simulator = functools.partial(
            generate_synthetic_musr_data,
            center=[0., 0.], curvature=[5., 10.],
            bounds=[(-1., 1.), (-1., 1.)],
            delta_us_inv=.8, rng=numpy.random.default_rng(2026),
        )
        snapshot = simulator([.2, -.4])

    The expected rate is ``counts_per_frame`` at the centre and
    ``counts_per_frame * edge_rate_fraction`` on *every parameter-space face*.
    ``curvature`` affects the KT sample/background mixture, while the bounds
    and edge fraction control the rate profile. This keeps them independently
    configurable. The *true* A(0) is identical at every coordinate.

    ``time_bins`` are bin CENTRES in ns; ``time_bin_edges`` includes the
    left boundary at 0 ns and the right boundary at 32000 ns by default.
    """
    x = np.asarray(coordinate, dtype=float)
    c = np.asarray(center, dtype=float)
    k = np.asarray(curvature, dtype=float)
    if x.ndim != 1 or x.size == 0 or c.shape != x.shape or k.shape != x.shape:
        raise ValueError("coordinate, center and curvature must all have the same nonzero 1-D shape")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(c)) and np.all(np.isfinite(k))):
        raise ValueError("coordinate, center and curvature must be finite")
    if np.any(k < 0):
        raise ValueError("curvature values must be >= 0")
    if not np.isfinite(initial_asymmetry) or abs(initial_asymmetry) >= 1:
        raise ValueError("initial_asymmetry must have absolute value < 1")
    if not np.isfinite(muon_lifetime_us) or muon_lifetime_us <= 0:
        raise ValueError("muon_lifetime_us must be finite and > 0")
    if not isinstance(n_time_bins, int) or isinstance(n_time_bins, bool) or n_time_bins < 2:
        raise ValueError("n_time_bins must be an integer >= 2")
    if not np.isfinite(max_time_us) or max_time_us <= 0:
        raise ValueError("max_time_us must be finite and > 0")
    if rng is not None and seed is not None:
        raise ValueError("Pass rng OR seed, not both")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    generator = rng if rng is not None else np.random.default_rng(seed)

    rate_fraction = spatial_rate_fraction(
        x, center=c, bounds=bounds, edge_rate_fraction=edge_rate_fraction,
    )
    expected_counts_per_frame = float(counts_per_frame) * rate_fraction
    frames, wall_seconds = synthetic_good_frames_and_seconds(
        duration, counts_per_frame=expected_counts_per_frame,
        good_frames_per_second=good_frames_per_second,
    )

    detector_groups = grouping if grouping is not None else musr_longitudinal_grouping()
    spectra = tuple(sorted(set(detector_groups.forward) | set(detector_groups.backward)))
    indices = {spectrum: row for row, spectrum in enumerate(spectra)}
    forward_rows = [indices[s] for s in detector_groups.forward]
    backward_rows = [indices[s] for s in detector_groups.backward]
    n_forward, n_backward = len(forward_rows), len(backward_rows)

    dt_us = float(max_time_us) / n_time_bins
    time_bin_edges_ns = np.linspace(0.0, max_time_us * 1000, n_time_bins + 1)
    time_bin_centres_ns = (time_bin_edges_ns[:-1] + time_bin_edges_ns[1:]) / 2.0
    time_us = time_bin_centres_ns / 1000.0
    weights = np.exp(-time_us / muon_lifetime_us)
    weights /= weights.sum()

    kt_fraction = float(np.exp(-np.sum(k * np.square(x - c))))
    kt = static_gaussian_kubo_toyabe(time_us, delta_us_inv)
    polarisation = kt_fraction * kt + (1 - kt_fraction)  # flat background P(t)=1
    asymmetry = float(initial_asymmetry) * polarisation

    expected_per_time_bin = frames * expected_counts_per_frame * weights
    expected_forward = expected_per_time_bin * (1 + asymmetry) / 2
    expected_backward = expected_per_time_bin * (1 - asymmetry) / 2
    expected = np.zeros((len(spectra), n_time_bins), dtype=float)
    expected[forward_rows, :] = expected_forward / n_forward
    expected[backward_rows, :] = expected_backward / n_backward

    # Poisson variance is the bin expectation; the bin standard deviation
    # falls as exp(-t/(2*tau_mu)). Use sqrt(measured) as the estimator.
    measured = generator.poisson(expected)
    forward = measured[forward_rows, :].sum(axis=0)
    backward = measured[backward_rows, :].sum(axis=0)
    total = int(measured.sum())

    return {
        "spectrum_numbers": np.asarray(spectra, dtype=np.int64),
        "counts_by_spectrum": measured,
        "count_std_by_spectrum": np.sqrt(measured.astype(float)),
        "forward_counts": forward,
        "backward_counts": backward,
        "total_counts": float(total),
        "good_frames": frames,
        "events_millions": total / 1e6,
        "wall_seconds": wall_seconds,
        "dae_run_duration_seconds": wall_seconds,  # simulated equivalent, NOT a DAE reading
        "time_bins": time_bin_centres_ns,
        "time_bin_edges": time_bin_edges_ns,
        "time_bin_unit": "ns",
        "time_bin_width_ns": dt_us * 1000.0,
        "synthetic": True,
        "sample_fraction": kt_fraction,
        "rate_fraction": rate_fraction,
        "expected_counts_per_frame": expected_counts_per_frame,
        "expected_polarisation": polarisation,
        "expected_asymmetry": asymmetry,
        "expected_counts_by_spectrum": expected,
        "expected_total_counts": float(expected_per_time_bin.sum()),
        "ideal_initial_asymmetry": float(initial_asymmetry),
        "coordinate": tuple(float(y) for y in x),
    }
