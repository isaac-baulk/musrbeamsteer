"""Simulator test; no live hardware or Bluesky installation needed."""
import sys
import types
import numpy as np

from ..blueskysteercore.bluesky_integration import beam_steering_plan


class FakeMotor:
    def __init__(self, value):
        self.value = float(value)


def fake_stubs():
    def rd(motor):
        if False:
            yield
        return motor.value

    def mv(*args):
        for motor, value in zip(args[::2], args[1::2]):
            motor.value = value
        if False:
            yield

    def sleep(seconds):
        if False:
            yield

    return types.SimpleNamespace(rd=rd, mv=mv, sleep=sleep)


def consume(generator):
    try:
        while True:
            next(generator)
    except StopIteration as end:
        return end.value


def test_plan_acquires_and_returns_result(monkeypatch):
    bluesky = types.ModuleType('bluesky')
    bluesky.__path__ = []
    stubs = fake_stubs()
    bluesky.plan_stubs = stubs
    monkeypatch.setitem(sys.modules, 'bluesky', bluesky)
    monkeypatch.setitem(sys.modules, 'bluesky.plan_stubs', stubs)
    motor = FakeMotor(0.2)
    points = []

    def acquire(coordinate):
        points.append(float(coordinate[0]))
        if False:
            yield
        return {'x': float(coordinate[0]), 'y': (float(coordinate[0]) - 0.1) ** 2 + 0.001 * len(points)}

    def metrics(measurements, first_set=False):
        return [m['y'] for m in measurements]

    result = consume(beam_steering_plan(
        [motor], [(-1, 1)], acquire, metrics,
        initial_coordinates=[(0.1,), (-0.5,)],
        initial_repeats=2, max_measurements=3, precisions=[2],
        verbose=False,
    ))
    assert len(points) == 3
    assert result.n_measurements == 3
    assert np.isclose(motor.value, 0.1)


def test_restores_on_failed_acquisition(monkeypatch):
    import pytest
    bluesky = types.ModuleType('bluesky')
    bluesky.__path__ = []
    stubs = fake_stubs()
    bluesky.plan_stubs = stubs
    monkeypatch.setitem(sys.modules, 'bluesky', bluesky)
    monkeypatch.setitem(sys.modules, 'bluesky.plan_stubs', stubs)
    motor = FakeMotor(0.7)

    def acquire(coordinate):
        if False:
            yield
        raise RuntimeError('DAQ failure')

    with pytest.raises(RuntimeError, match='DAQ failure'):
        consume(beam_steering_plan([motor], [(-1, 1)], acquire, lambda a,b: [],
            initial_coordinates=[(0.1,)], verbose=False))
    assert np.isclose(motor.value, 0.7)
