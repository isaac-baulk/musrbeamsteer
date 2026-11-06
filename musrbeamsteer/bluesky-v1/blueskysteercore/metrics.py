"""Reusable minimisation metrics; smaller is always better.

Measurements are mappings containing event data or a measurement identifier.
Independent metrics cache by a stable measurement ID, not by coordinates:
repeats at one position are distinct observations for noise estimation.
Global metrics (e.g. PCA) recalculate the entire dataset without caching.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any
import numpy as np


def _scalar(value: Any, label: str) -> float:
    a = np.asarray(value, dtype=float)
    if a.size != 1:
        raise ValueError(f"{label} must be scalar, got shape {a.shape}")
    result = float(a.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _field(record: Mapping[str, Any], name: str) -> Any:
    data = record.get('data', record)
    if name in data:
        return data[name]
    raise KeyError(f"Measurement {record.get('uid', '?')} lacks '{name}'; available: {list(data)}")


class IndependentMetric:
    """Cache one independently evaluated score per measurement UID.

    Records are stored per extractor instance. `cache_info` and `cache_clear`
    expose functools.lru_cache controls. Avoid reusing UIDs for changed data.
    """
    changes_with_dataset = False

    def __init__(self, calculate: Callable[[Mapping[str, Any]], float], *, maximize: bool = False,
                 cache_size: int = 2048):
        self.calculate = calculate
        self.maximize = maximize
        self._records: dict[str, Mapping[str, Any]] = {}

        @lru_cache(maxsize=cache_size)
        def cached(uid: str) -> float:
            score = _scalar(self.calculate(self._records[uid]), 'metric')
            return -score if self.maximize else score

        self._cached = cached

    def __call__(self, measurements: Sequence[Mapping[str, Any]], first_set: bool = False) -> list[float]:
        result = []
        for record in measurements:
            uid = record.get('uid')
            if uid is None:
                raise ValueError('IndependentMetric requires a unique measurement uid')
            uid = str(uid)
            if uid in self._records and self._records[uid] is not record:
                # IDs identify immutable acquisitions; do not silently reuse an ID.
                raise ValueError(f'Measurement UID {uid!r} reused with a different record')
            self._records[uid] = record
            result.append(self._cached(uid))
        return result

    def cache_info(self):
        return self._cached.cache_info()

    def cache_clear(self):
        self._cached.cache_clear()
        self._records.clear()


class GlobalMetric:
    """Dataset-dependent metric such as PCA; deliberately never cached."""
    changes_with_dataset = True

    def __init__(self, calculate: Callable[..., Sequence[float]]):
        self.calculate = calculate

    def __call__(self, measurements, first_set=False):
        return list(self.calculate(measurements, first_set))


def _asymmetry_window(record: Mapping[str, Any], bins: int, start_bin: int,
                      time_window: tuple[float, float] | None) -> np.ndarray | slice:
    """Time window for initial asymmetry, in bin indices OR native DAE time units.

    For MuSR the DAE normally supplies time in ns. A time_window=(t0,t1)
    uses `time_bins` from the live snapshot. A bin-based window operates on
    the raw, unaligned time channels, so set start_bin past time zero.
    """
    if time_window is not None:
        lo, hi = time_window
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError('time_window must contain increasing finite bounds')
        t = np.asarray(_field(record, 'time_bins'), dtype=float).reshape(-1)
        mask = (t >= lo) & (t < hi)
        if not mask.any():
            raise ValueError('asymmetry time window selects no DAE bins')
        return mask
    if bins < 1 or start_bin < 0:
        raise ValueError('bins >= 1 and start_bin >= 0 required')
    return slice(start_bin, start_bin + bins)


def _asymmetry_components(record: Mapping[str, Any], *, forward_field: str,
                          backward_field: str, bins: int, start_bin: int,
                          time_window: tuple[float, float] | None, alpha: float) -> tuple[float, float, float]:
    selection = _asymmetry_window(record, bins, start_bin, time_window)
    f_array = np.asarray(_field(record, forward_field), dtype=float).reshape(-1)
    b_array = np.asarray(_field(record, backward_field), dtype=float).reshape(-1)
    if f_array.size != b_array.size:
        raise ValueError('forward and backward detector arrays have different lengths')
    if isinstance(selection, np.ndarray) and selection.size != f_array.size:
        raise ValueError('time_bins length does not match detector arrays')
    f = float(f_array[selection].sum())
    b = float(b_array[selection].sum())
    denom = f + alpha * b
    if denom <= 0:
        raise ValueError('asymmetry is undefined for zero counts in chosen time window')
    return f, b, float((f - alpha * b) / denom)


def count_rate_metric(*, counts_field='total_counts', frames_field='good_frames',
                      per='good_frame', seconds_field='wall_seconds',
                      maximize=True) -> IndependentMetric:
    """Integrated counts / good frames (default) or / elapsed wall seconds.

    For `per='second'`, this is wall-clock-normalised count rate, not a
    dead-time/veto corrected rate. Good frames are generally preferable for
    beam tuning and for comparisons across variable beam interruptions.
    """
    if per not in ('good_frame', 'second'):
        raise ValueError("per must be 'good_frame' or 'second'")

    def score(record):
        name = frames_field if per == 'good_frame' else seconds_field
        exposure = _scalar(_field(record, name), name)
        if exposure <= 0:
            raise ValueError(f'{name} must be positive')
        return _scalar(_field(record, counts_field), counts_field) / exposure

    return IndependentMetric(score, maximize=maximize)


def initial_asymmetry_metric(*, forward_field='forward_counts', backward_field='backward_counts',
                             alpha=1.0, bins=1, start_bin=0,
                             time_window: tuple[float, float] | None = None,
                             maximize=True) -> IndependentMetric:
    """(F-alpha B)/(F+alpha B) integrated over selected raw time bins.

    Supports `start_bin` and `bins` (slice) or `time_window` in DAE time units.
    This is an EARLY-TIME ESTIMATE, not the fitted A_0 of a muon oscillator.
    Detector t0, efficiency, deadtime, and backgrounds require calibration.
    """
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('alpha must be finite and positive')

    def score(record):
        _, _, a = _asymmetry_components(
            record, forward_field=forward_field, backward_field=backward_field,
            bins=bins, start_bin=start_bin, time_window=time_window, alpha=alpha,
        )
        return a

    return IndependentMetric(score, maximize=maximize)


def asymmetry_squared_count_rate_metric(*, forward_field='forward_counts',
                                        backward_field='backward_counts', alpha=1.0,
                                        bins=1, start_bin=0,
                                        time_window: tuple[float, float] | None = None,
                                        rate_counts_field='total_counts',
                                        frames_field='good_frames', maximize=True) -> IndependentMetric:
    """A_early**2 * (total integrated detector counts / good frames).

    The asymmetry time window and the count-rate integration range are
    intentionally separate. `rate_counts_field='total_counts'` uses the full
    snapshot spectrum; choose a different count field if desired.
    """
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('alpha must be finite and positive')

    def score(record):
        _, _, a = _asymmetry_components(
            record, forward_field=forward_field, backward_field=backward_field,
            bins=bins, start_bin=start_bin, time_window=time_window, alpha=alpha,
        )
        frames = _scalar(_field(record, frames_field), frames_field)
        if frames <= 0:
            raise ValueError('positive good frames required')
        rate = _scalar(_field(record, rate_counts_field), rate_counts_field) / frames
        return a*a*rate

    return IndependentMetric(score, maximize=maximize)


def legacy_metrics(callback: Callable[..., Sequence[float]]) -> GlobalMetric:
    """Adapt dataset-dependent PCA extractors that take the full history."""
    return GlobalMetric(callback)


