"""Numerical checks for the new direct-histogram muon metric implementations."""
from __future__ import annotations

import numpy as np
import pytest

from ..blueskysteercore.musr_metrics import integrated_asymmetry_metric, pca_asymmetry_metric


def record(uid, forward, backward, *, times=None):
    if times is None:
        times = np.arange(len(forward), dtype=float) * 16 + 8
    return {
        'uid': uid,
        'data': {
            'forward_counts': np.asarray(forward, dtype=float),
            'backward_counts': np.asarray(backward, dtype=float),
            'time_bins': np.asarray(times, dtype=float),
        },
    }


def test_integrated_matches_existing_inverse_variance_weighting():
    # For alpha=1 and F+B=100, sigma(A)^2=(1-A^2)/100.
    a = record('a', [80, 60], [20, 40])
    metric = integrated_asymmetry_metric(rebin=1, time_window_ns=(0, 100))
    assert metric([a]) == [pytest.approx(0.44)]
    assert metric([a]) == [pytest.approx(0.44)]
    assert metric.cache_info().hits == 1


def test_rebin_counts_before_asymmetry_and_weighting():
    a = record('a', [80, 60], [20, 40])
    # One rebinned group: F=140, B=60, A=.4
    metric = integrated_asymmetry_metric(rebin=2, time_window_ns=(0, 100))
    assert metric([a])[0] == pytest.approx(.4)


def test_bad_window_and_no_counts_are_rejected():
    a = record('a', [0, 0], [0, 0])
    with pytest.raises(ValueError, match='No valid'):
        integrated_asymmetry_metric(rebin=1)([a])
    with pytest.raises(ValueError, match='selects no'):
        integrated_asymmetry_metric(time_window_ns=(1000, 2000))([a])
    with pytest.raises(ValueError, match='alpha'):
        integrated_asymmetry_metric(alpha=0)
    with pytest.raises(ValueError, match='rebin'):
        integrated_asymmetry_metric(rebin=0)([a])


def test_pca_requires_common_time_grid():
    a = record('a', [80, 60], [20, 40])
    b = record('b', [80, 60], [20, 40], times=[10, 30])
    metric = pca_asymmetry_metric(rebin=1)
    with pytest.raises(ValueError, match='same time axis'):
        metric([a, b])


def test_pca_global_recomputes_first_scores_when_dataset_changes():
    a = record('a', [51, 80, 70], [49, 20, 30])
    b = record('b', [52, 60, 50], [48, 40, 50])
    c = record('c', [51, 65, 45], [49, 35, 55])
    metric = pca_asymmetry_metric(rebin=1)
    first = metric([a, b])
    all_three = metric([a, b, c])
    assert len(first) == 2
    assert len(all_three) == 3
    assert all(np.isfinite(all_three))
    assert metric.changes_with_dataset
    assert not np.allclose(first, all_three[:2])
