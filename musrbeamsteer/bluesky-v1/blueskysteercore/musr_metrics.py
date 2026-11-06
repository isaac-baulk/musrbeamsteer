"""Muon time-histogram reductions shared by synthetic and live ISIS DAE.

Both backends return the same forward/backward time histograms and ``time_bins``
(bin centres, ns). No NeXus/Mantid intermediate file is required.

* Integrated asymmetry is the inverse-variance-weighted average of the
  time-binned (F-alpha B)/(F+alpha B), consistent with the original project.
  It is an independent metric, and is cached per unique measurement UID.
* PCA works directly on time-dependent asymmetry curves. The leading PC1
  score is oriented to correlate *positively* with integrated asymmetry so that
  minimisation can find reduced polarisation. PCA depends on every record and
  must recalculate historical scores whenever a new measurement is acquired.

These basic reductions do not perform t0, background, dead-time or detector
calibration. Configure alpha and select an appropriate post-t0 time range when
using a real DAE.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .metrics import IndependentMetric, GlobalMetric, _field


def _time_binned_asymmetry(
    record: Mapping[str, Any],
    *,
    alpha: float,
    time_window_ns: tuple[float, float],
    rebin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return time (ns), A, Poisson sigma(A), and summed counts per group."""
    f = np.asarray(_field(record, 'forward_counts'), dtype=float).reshape(-1)
    b = np.asarray(_field(record, 'backward_counts'), dtype=float).reshape(-1)
    t = np.asarray(_field(record, 'time_bins'), dtype=float).reshape(-1)
    if f.size != b.size or t.size != f.size or not f.size:
        raise ValueError('time_bins and forward/backward counts must have matching nonzero lengths')
    if not (np.isfinite(f).all() and np.isfinite(b).all() and np.isfinite(t).all()):
        raise ValueError('non-finite time or detector counts')
    if np.any(f < 0) or np.any(b < 0) or np.any(np.diff(t) <= 0):
        raise ValueError('counts must be nonnegative and time bins strictly increasing')

    lo, hi = time_window_ns
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError('time_window_ns must have finite, increasing endpoints')
    if not isinstance(rebin, int) or isinstance(rebin, bool) or rebin < 1:
        raise ValueError('rebin must be an integer >= 1')
    mask = (t >= lo) & (t < hi)
    if not mask.any():
        raise ValueError('time_window_ns selects no time bins')
    t, f, b = t[mask], f[mask], b[mask]
    if rebin > 1:
        starts = np.arange(0, t.size, rebin)
        sizes = np.diff(np.r_[starts, t.size])
        t = np.add.reduceat(t, starts) / sizes
        f = np.add.reduceat(f, starts)
        b = np.add.reduceat(b, starts)

    denom = f + alpha * b
    with np.errstate(divide='ignore', invalid='ignore'):
        asymmetry = (f - alpha * b) / denom
        # Independent Poisson forward and backward counts. Propagate the
        # (F-alpha*B)/(F+alpha*B) uncertainty via partial derivatives.
        variance = 4.0 * alpha**2 * f * b * (f + b) / denom**4
    sigma = np.sqrt(variance)
    return t, asymmetry, sigma, f + b


def integrated_asymmetry_metric(
    *,
    alpha: float = 1.0,
    time_window_ns: tuple[float, float] = (0.0, 8000.0),
    rebin: int = 8,
    min_bin_counts: float = 10.0,
    maximize: bool = False,
) -> IndependentMetric:
    """Inverse-variance-weighted integrated asymmetry across a time window.

    The weighted integral is ``sum(A_i/sigma_i**2) / sum(1/sigma_i**2)``.
    It equals an inverse-variance-weighted *average*, not a raw sum, so it is
    independent of acquisition duration in expectation. In the static KT +
    flat-background test the positive asymmetry is smaller at the KT centre,
    so ``maximize=False`` finds that centre.

    The ``time_window_ns`` applies to time-bin centres. ``rebin`` combines
    adjacent time bins by summing counts BEFORE computing A and its error.
    """
    _validate_metric_options(alpha, min_bin_counts)

    def calculate(record: Mapping[str, Any]) -> float:
        _, asymmetry, sigma, totals = _time_binned_asymmetry(
            record, alpha=alpha, time_window_ns=time_window_ns, rebin=rebin,
        )
        valid = (
            np.isfinite(asymmetry) & np.isfinite(sigma) & (sigma > 0)
            & (totals >= min_bin_counts)
        )
        if not np.any(valid):
            raise ValueError('No valid time bins for integrated asymmetry')
        weights = 1.0 / np.square(sigma[valid])
        return float(np.average(asymmetry[valid], weights=weights))

    return IndependentMetric(calculate, maximize=maximize)

def asym_squared_count_rate_metric(
    *,
    alpha: float = 1.0,
    time_window_ns: tuple[float, float] = (0.0, 8000.0),
    rebin: int = 8,
    min_bin_counts: float = 10.0,
    maximize: bool = True,
) -> IndependentMetric:
    """Inverse-variance-weighted integrated asymmetry squared times count rate.

    This metric is the integrated asymmetry squared multiplied by the total
    counts in the selected time window, divided by the exposure (good frames).
    It is a proxy for polarisation squared times count rate, which is a
    common figure of merit for muon experiments.
    """
    _validate_metric_options(alpha, min_bin_counts)

    def calculate(record: Mapping[str, Any]) -> float:
        _, asymmetry, sigma, totals = _time_binned_asymmetry(
            record, alpha=alpha, time_window_ns=time_window_ns, rebin=rebin,
        )
        valid = (
            np.isfinite(asymmetry) & np.isfinite(sigma) & (sigma > 0)
            & (totals >= min_bin_counts)
        )
        if not np.any(valid):
            raise ValueError('No valid time bins for asymmetry squared count rate')
        weights = 1.0 / np.square(sigma[valid])
        integrated_asymmetry = float(np.average(asymmetry[valid], weights=weights))
        total_counts = float(totals[valid].sum())
        exposure = float(_field(record, 'good_frames'))
        if exposure <= 0:
            raise ValueError('good_frames must be positive for asymmetry squared count rate')
        return (integrated_asymmetry ** 2) * (total_counts / exposure)

    return IndependentMetric(calculate, maximize=maximize)


def pca_asymmetry_metric(
    *,
    alpha: float = 1.0,
    time_window_ns: tuple[float, float] = (0.0, 8000.0),
    rebin: int = 8,
    min_bin_counts: float = 10.0,
) -> GlobalMetric:
    """Leading principal-component scores of time-resolved asymmetry curves.

    The matrix rows are measured *asymmetry*, not unnormalised raw counts,
    thus the synthetic 10% edge count rate affects statistical uncertainty
    but does not introduce a trivial amplitude feature into the PCA. Bins
    must be valid for *every* record, so the same features are compared.

    SVD signs are arbitrary; PC1 is oriented so that its inverse-variance-
    weighted time integral is POSITIVE. Smaller PC1 score then means smaller
    weighted asymmetry, matching ``integrated_asymmetry_metric`` in a dominant
    one-component signal. This is a live-data PCA, without legacy NeXus files
    or the historical TT_PCA extra preprocessing.

    PCA scores depend on the full set of measurements. ``GlobalMetric``
    intentionally never caches them, even if the new coordinate is a repeat.
    """
    _validate_metric_options(alpha, min_bin_counts)

    def calculate(measurements: Sequence[Mapping[str, Any]], first_set: bool = False) -> list[float]:
        if not measurements:
            return []
        time_grid: np.ndarray | None = None
        curves: list[np.ndarray] = []
        sigmas: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for record in measurements:
            t, a, sigma, total = _time_binned_asymmetry(
                record, alpha=alpha, time_window_ns=time_window_ns, rebin=rebin,
            )
            if time_grid is None:
                time_grid = t
            elif t.shape != time_grid.shape or not np.allclose(t, time_grid, rtol=0, atol=1e-6):
                raise ValueError('PCA requires the same time axis for every measurement')
            curves.append(a)
            sigmas.append(sigma)
            counts.append(total)

        mat = np.stack(curves)
        error = np.stack(sigmas)
        total = np.stack(counts)
        mask = np.all(
            np.isfinite(mat) & np.isfinite(error) & (error > 0)
            & (total >= min_bin_counts), axis=0,
        )
        if not np.any(mask):
            raise ValueError('No common valid time bins for PCA')
        y = mat[:, mask]
        if y.shape[0] == 1:
            return [0.0]
        demeaned = y - y.mean(axis=0, keepdims=True)
        _, singular_values, components = np.linalg.svd(demeaned, full_matrices=False)
        # First score is X @ first right singular vector, directly in the
        # asymmetry units of the dataset, not weighted by exposure.
        pc1 = demeaned @ components[0]
        # Orient PC1 with the integrated asymmetry; PCA itself is sign-free.
        weights = np.mean(1.0 / np.square(error[:, mask]), axis=0)
        if float(np.dot(components[0], weights)) < 0:
            pc1 = -pc1
        if not np.isfinite(pc1).all():
            raise ValueError('PCA yielded non-finite scores')
        return [float(value) for value in pc1]

    return GlobalMetric(calculate)

def _validate_metric_options(alpha: float, min_bin_counts: float) -> None:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    if not np.isfinite(min_bin_counts) or min_bin_counts <= 0:
        raise ValueError('min_bin_counts must be finite and positive')
