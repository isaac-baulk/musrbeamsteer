"""Fake EPICS/Ophyd Async integration: signed setpoint and readback only."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import sys
import types

import pytest

from ..blueskysteercore.positioners import make_setpoint_readback_axis


def test_signed_setpoint_readback_positioner_without_polarity_or_power(monkeypatch):
    written = []
    checked = []
    read_pvs = []

    class FakeSignal:
        def __init__(self, pv): self.pv = pv; self.value = 0.

        async def set(self, target, timeout=None):
            written.append((self.pv, target, timeout))
            readback.value = target + 0.002

    readback = FakeSignal('IN:MUSR:TEST:RBV')

    class StandardReadable:
        def __init__(self, name): self.name = name

        @contextmanager
        def add_children_as_readables(self, fmt):
            yield

    class AsyncStatus:
        @staticmethod
        def wrap(fn): return fn

    async def wait_for_value(signal, matcher, timeout):
        checked.append((signal.pv, timeout))
        if not matcher(signal.value):
            raise TimeoutError('readback did not settle')

    core = types.ModuleType('ophyd_async.core')
    core.StandardReadable = StandardReadable
    core.AsyncStatus = AsyncStatus
    core.HintedSignal = object()
    core.wait_for_value = wait_for_value
    epics = types.ModuleType('ophyd_async.epics')
    epics.__path__ = []
    signals = types.ModuleType('ophyd_async.epics.signal')
    signals.epics_signal_w = lambda dtype, pv: FakeSignal(pv)
    signals.epics_signal_r = lambda dtype, pv: (read_pvs.append(pv), readback)[1]
    pkg = types.ModuleType('ophyd_async')
    pkg.__path__ = []
    pkg.core = core
    pkg.epics = epics
    epics.signal = signals
    for name, obj in [('ophyd_async', pkg), ('ophyd_async.core', core),
                      ('ophyd_async.epics', epics), ('ophyd_async.epics.signal', signals)]:
        monkeypatch.setitem(sys.modules, name, obj)

    axis = make_setpoint_readback_axis(
        name='MUSR_STEER_TEST', setpoint_pv='IN:MUSR:TEST:SP',
        readback_pv='IN:MUSR:TEST:RBV', tolerance=0.01,
        timeout=4, limits=(-2., 2.),
    )
    asyncio.run(axis.set(-1.2))
    assert written == [('ca://IN:MUSR:TEST:SP', -1.2, 4)]
    assert read_pvs == ['ca://IN:MUSR:TEST:RBV']
    assert checked == [('IN:MUSR:TEST:RBV', 4)]
    assert not hasattr(axis, 'polarity')
    assert not hasattr(axis, 'power')
    with pytest.raises(ValueError, match='limits'):
        asyncio.run(axis.set(3.0))
    assert len(written) == 1


def test_positioner_validates_parameters_without_ophyd_import():
    with pytest.raises(ValueError, match='required'):
        make_setpoint_readback_axis(name='', setpoint_pv='SP', readback_pv='RBV', tolerance=0.01)


def test_transport_defaults_to_ca_but_respects_explicit_prefixes():
    from ..blueskysteercore.positioners import _epics_pv
    assert _epics_pv("IN:MUSR:TEST:SP", "ca") == "ca://IN:MUSR:TEST:SP"
    assert _epics_pv("IN:MUSR:TEST:SP", "pva") == "pva://IN:MUSR:TEST:SP"
    assert _epics_pv("pva://IN:MUSR:TEST:SP", "ca") == "pva://IN:MUSR:TEST:SP"
    assert _epics_pv("ca://IN:MUSR:TEST:SP", "pva") == "ca://IN:MUSR:TEST:SP"
    with pytest.raises(ValueError, match="Unsupported EPICS"):
        _epics_pv("http://INVALID", "ca")
