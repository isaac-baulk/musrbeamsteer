"""Contract test for the factory, using tiny stubs of the public ISIS API."""
from __future__ import annotations
import asyncio
import sys
import types

import numpy as np

from ..blueskysteercore.isis_dae import AcquisitionDuration, make_live_run_per_point_dae


class Scalar:
    def __init__(self, value):
        self.value = value
    async def get_value(self):
        return self.value


def test_construct_and_trigger_live_dae_using_published_public_api(monkeypatch):
    pkg = types.ModuleType('ibex_bluesky_core'); pkg.__path__ = []
    devices = types.ModuleType('ibex_bluesky_core.devices'); devices.__path__ = []
    dae_mod = types.ModuleType('ibex_bluesky_core.devices.dae')
    simple = types.ModuleType('ibex_bluesky_core.devices.simpledae')
    utils = types.ModuleType('ibex_bluesky_core.utils')
    ophyd = types.ModuleType('ophyd_async'); ophyd.__path__ = []
    core = types.ModuleType('ophyd_async.core')

    class RunstateEnum:
        SETUP='SETUP'; PAUSED='PAUSED'

    class DaeSpectra:
        def __init__(self, dae_prefix, *, spectra, period):
            assert dae_prefix == 'IN:MUSR:DAE:'
            assert spectra == 1 and period == 1
        async def read_tof(self):
            return np.array([100.,110.,120.])

    def soft_signal_r_and_setter(type_, initial):
        scalar = Scalar(initial)
        def setter(value): scalar.value=value
        return scalar, setter

    class AsyncStatus:
        @staticmethod
        def wrap(fn): return fn

    async def wait_for_value(signal, state, timeout):
        assert await signal.get_value() == state

    class StubController:
        def __init__(self, save_run):
            assert save_run is True
            self.run_number=Scalar(42)
        async def start_counting(self, dae):
            dae.log.append('BEGIN')
        async def stop_counting(self, dae):
            dae.log.append('END')

    class StubWaiter:
        def __init__(self, value):
            assert value == 100
        async def wait(self, dae):
            dae.log.append('WAIT')

    class Reducer: pass

    class SimpleDae:
        def __init__(self, *, prefix, name, controller, waiter, reducer):
            assert prefix == 'IN:MUSR:'
            self.name=name
            self.controller=controller
            self.waiter=waiter
            self.reducer=reducer
            self.log=[]
            self.number_of_periods=types.SimpleNamespace(signal=Scalar(1))
            self.num_spectra=Scalar(64)
            self.run_state=Scalar(RunstateEnum.SETUP)
            self.period=types.SimpleNamespace(
                good_frames=Scalar(100),
                run_duration=Scalar(5),
            )
            self.m_events=Scalar(0.5)
            self.run_duration=Scalar(5)
            async def pause_run():
                self.log.append('PAUSE')
                self.run_state.value=RunstateEnum.PAUSED
            self.controls=types.SimpleNamespace(pause_run=types.SimpleNamespace(
                trigger=lambda timeout: pause_run()))
        def add_readables(self, *, devices):
            assert len(devices) == 3
        async def trigger_and_get_specdata(self, indices):
            self.log.append('SNAPSHOT')
            return np.ones((len(indices),3), dtype=np.int32)

    dae_mod.DaeSpectra=DaeSpectra
    dae_mod.RunstateEnum=RunstateEnum
    simple.SimpleDae=SimpleDae
    simple.Reducer=Reducer
    simple.RunPerPointController=StubController
    simple.PeriodGoodFramesWaiter=StubWaiter
    simple.MEventsWaiter=StubWaiter
    simple.TimeWaiter=StubWaiter
    utils.get_pv_prefix=lambda: 'IN:MUSR:'
    core.AsyncStatus=AsyncStatus
    core.soft_signal_r_and_setter=soft_signal_r_and_setter
    core.wait_for_value=wait_for_value
    for name, module in (
        ('ibex_bluesky_core',pkg),('ibex_bluesky_core.devices',devices),
        ('ibex_bluesky_core.devices.dae', dae_mod),
        ('ibex_bluesky_core.devices.simpledae',simple),
        ('ibex_bluesky_core.utils',utils),
        ('ophyd_async',ophyd),('ophyd_async.core',core),
    ):
        monkeypatch.setitem(sys.modules,name,module)

    dae=make_live_run_per_point_dae(
        duration=AcquisitionDuration.good_frames(100),
        spectra=[1,2],
    )
    asyncio.run(dae.trigger())
    assert dae.log == ['BEGIN','WAIT','PAUSE','SNAPSHOT','END']
    assert dae.last_snapshot['run_number'] == 42
    assert dae.last_snapshot['total_counts'] == 6
    assert dae.snapshot_counts.value == 6
    assert dae.snapshot_frames.value == 100
    assert dae.snapshot_run_number.value == 42
