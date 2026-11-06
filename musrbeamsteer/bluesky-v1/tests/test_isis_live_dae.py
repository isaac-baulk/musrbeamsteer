"""Pure mocked unit tests: no ISIS installation, DAE, EPICS or Bluesky needed."""
from __future__ import annotations

import asyncio
import sys
import types

import numpy as np
import pytest

from ..blueskysteercore.isis_dae import (
    AcquisitionDuration, DetectorGrouping, LiveSpectrumSnapshotter,
    acquire_snapshot_before_save, make_duration_waiter, musr_longitudinal_grouping,
)
from ..blueskysteercore.metrics import (
    initial_asymmetry_metric, asymmetry_squared_count_rate_metric, count_rate_metric,
)


class Signal:
    def __init__(self, value):
        self.value = value

    async def get_value(self):
        return self.value


class FakeDAE:
    def __init__(self, *, n_bins=4):
        self.calls = []
        self.controller = types.SimpleNamespace(
            run_number=Signal(12345), start_counting=self.start, stop_counting=self.end
        )
        self.waiter = types.SimpleNamespace(wait=self.wait)
        self.period = types.SimpleNamespace(
            good_frames=Signal(100),
            run_duration=Signal(10),
        )
        self.m_events = Signal(0.5)
        self.run_duration = Signal(10)
        self.live_time_axis = types.SimpleNamespace(read_tof=self.read_tof)
        self.n_bins = n_bins
        self.fail_snapshot = False
        self.fail_wait = False

    async def start(self, dae):
        self.calls.append("BEGIN")

    async def wait(self, dae):
        self.calls.append("WAIT")
        if self.fail_wait:
            raise RuntimeError("beam disappeared")

    async def pause(self, dae):
        self.calls.append("PAUSE")

    async def trigger_and_get_specdata(self, detectors):
        self.calls.append("LIVE_UPDATE_AND_READ")
        if self.fail_snapshot:
            raise RuntimeError("cannot read live DAE")
        return np.vstack([np.arange(1, self.n_bins + 1) + i for i in range(len(detectors))])

    async def read_tof(self):
        self.calls.append("READ_TCB")
        return np.arange(self.n_bins) * 10

    async def end(self, dae):
        self.calls.append("END_SAVE")


def test_duration_values_and_conversions(monkeypatch):
    assert AcquisitionDuration.events(250_000).value == 250_000
    assert AcquisitionDuration.good_frames(500).unit.value == 'good_frames'
    assert AcquisitionDuration.seconds(0.25).unit.value == 'seconds'
    with pytest.raises(ValueError):
        AcquisitionDuration.events(-100)
    with pytest.raises(ValueError):
        AcquisitionDuration.events(12.5)

    pkg = types.ModuleType('ibex_bluesky_core')
    pkg.__path__ = []
    devices = types.ModuleType('ibex_bluesky_core.devices')
    devices.__path__ = []
    simpledae = types.ModuleType('ibex_bluesky_core.devices.simpledae')
    class Frames:
        def __init__(self, value): self.value = value
    class MEv:
        def __init__(self, value): self.value = value
    class Seconds:
        def __init__(self, *, seconds): self.seconds = seconds
    simpledae.PeriodGoodFramesWaiter = Frames
    simpledae.MEventsWaiter = MEv
    simpledae.TimeWaiter = Seconds
    monkeypatch.setitem(sys.modules, 'ibex_bluesky_core', pkg)
    monkeypatch.setitem(sys.modules, 'ibex_bluesky_core.devices', devices)
    monkeypatch.setitem(sys.modules, 'ibex_bluesky_core.devices.simpledae', simpledae)
    assert isinstance(make_duration_waiter(AcquisitionDuration.good_frames(500)), Frames)
    assert make_duration_waiter(AcquisitionDuration.events(250_000)).value == pytest.approx(0.25)
    assert make_duration_waiter(AcquisitionDuration.seconds(7.2)).seconds == 7.2


def test_musr_groups_not_argus():
    groups = musr_longitudinal_grouping()
    assert groups.forward == tuple(range(33,65))
    assert groups.backward == tuple(range(1,33))
    with pytest.raises(ValueError):
        DetectorGrouping((1,2), (2,3))
    with pytest.raises(ValueError):
        LiveSpectrumSnapshotter((1,2), grouping=groups)


def test_live_snapshot_precedes_end_and_yields_real_run_number():
    dae = FakeDAE()
    # Non-contiguous selection tests that F/B labels refer to actual spectra.
    grouping = DetectorGrouping((5, 9), (2, 3))
    snapshotter = LiveSpectrumSnapshotter((2, 3, 5, 9), grouping=grouping)
    result = asyncio.run(acquire_snapshot_before_save(
        dae, snapshotter, lambda d: d.pause(d), max_wait_seconds=60.0,
    ))
    assert dae.calls == [
        'BEGIN','WAIT','PAUSE','LIVE_UPDATE_AND_READ','READ_TCB','END_SAVE'
    ]
    assert result['run_number'] == 12345
    assert result['good_frames'] == 100
    assert result['counts_by_spectrum'].shape == (4,4)
    assert np.array_equal(result['backward_counts'], [3.,5.,7.,9.])
    assert np.array_equal(result['forward_counts'], [7.,9.,11.,13.])
    assert result['total_counts'] == result['counts_by_spectrum'].sum()
    assert result['time_bin_unit'] == 'ns'


def test_live_metric_with_window_and_total_rate():
    dae = FakeDAE()
    groups = DetectorGrouping((3, 4), (1, 2))
    snapshotter = LiveSpectrumSnapshotter((1,2,3,4), grouping=groups)
    snapshot = asyncio.run(snapshotter.capture(dae, wall_seconds=10.0))
    rec = {'uid':'a', 'data': snapshot}
    early = initial_asymmetry_metric(alpha=1.0, time_window=(10, 30))
    figure = asymmetry_squared_count_rate_metric(alpha=1.0, time_window=(10, 30))
    rate = count_rate_metric()
    assert early([rec])[0] < 0  # minimise negative asymmetry -> positive objective
    assert rate([rec])[0] == pytest.approx(-snapshot['total_counts']/100)
    assert figure([rec])[0] < 0
    early([rec]); assert early.cache_info().hits == 1


def test_failed_wait_and_failed_snapshot_both_end_run():
    for failure in ('fail_wait','fail_snapshot'):
        dae = FakeDAE()
        setattr(dae, failure, True)
        with pytest.raises(RuntimeError):
            asyncio.run(acquire_snapshot_before_save(
                dae, LiveSpectrumSnapshotter((1,2)), lambda d: d.pause(d),
                max_wait_seconds=60.0,
            ))
        assert dae.calls[-1] == 'END_SAVE'


def test_dead_beam_timeout_still_ends_run():
    dae = FakeDAE()
    async def never_returns(dae):
        await asyncio.sleep(100)
    dae.waiter = types.SimpleNamespace(wait=never_returns)
    with pytest.raises(TimeoutError):
        asyncio.run(acquire_snapshot_before_save(
            dae, LiveSpectrumSnapshotter((1,)), lambda d: d.pause(d),
            max_wait_seconds=0.005,
        ))
    assert dae.calls[-1] == 'END_SAVE'


def test_snapshot_empty_frames_fails_closed():
    dae = FakeDAE()
    dae.period.good_frames.value = 0
    with pytest.raises(ValueError, match='zero good frames'):
        asyncio.run(acquire_snapshot_before_save(
            dae, LiveSpectrumSnapshotter((1,)), lambda d: d.pause(d),
            max_wait_seconds=10,
        ))
    assert dae.calls[-1] == 'END_SAVE'
