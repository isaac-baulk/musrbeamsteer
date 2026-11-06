"""Live ISIS DAE run-per-point acquisition, independent of NeXus processing.

The hardware-facing factory in this module imports `ibex_bluesky_core` lazily.
The duration, snapshot and orchestration logic can be unit-tested without EPICS.

Crucial ordering for a run-per-point acquisition:

    BEGIN -> wait -> PAUSE -> UPDATE + snapshot spectra -> END (save)

The standard `SimpleDae` trigger ENDs the run *before* invoking the reducer;
this implementation changes that order so that the measurement is captured
from the LIVE DAE while it still owns the data. In RunPerPoint mode, this
does NOT avoid any time required by the DAE to end/save a run, only the subsequent wait for NeXus
availability before reduction. PeriodPerPoint mode is similar, but the DAE run is not ended until the device is unstaged, so
should be theoretically faster as START/END time is reduced.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from time import monotonic
from typing import Any

import numpy as np


class DurationUnit(str, Enum):
    GOOD_FRAMES = "good_frames"
    EVENTS = "events"
    SECONDS = "seconds"


@dataclass(frozen=True)
class AcquisitionDuration:
    """An acquisition target; 'events' denotes individual counts, not MEv."""

    unit: DurationUnit | str
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", DurationUnit(self.unit))
        if not np.isfinite(self.value) or self.value <= 0:
            raise ValueError("Acquisition duration must be finite and > 0")
        if self.unit in {DurationUnit.GOOD_FRAMES, DurationUnit.EVENTS}:
            if not float(self.value).is_integer():
                raise ValueError(f"{self.unit.value} must be an integer count")

    @classmethod
    def good_frames(cls, number: int) -> AcquisitionDuration:
        return cls(DurationUnit.GOOD_FRAMES, number)

    @classmethod
    def events(cls, number: int) -> AcquisitionDuration:
        return cls(DurationUnit.EVENTS, number)

    @classmethod
    def seconds(cls, number: float) -> AcquisitionDuration:
        return cls(DurationUnit.SECONDS, number)


def make_duration_waiter(duration: AcquisitionDuration) -> Any:
    """Create the actual IBEX waiter, converting events to *millions* of events.

    ISIS MEventsWaiter accepts MEv, whereas this API intentionally accepts
    raw event counts. For example, events(250_000) means 0.25 MEv.
    """
    from ibex_bluesky_core.devices.simpledae import (
        MEventsWaiter, PeriodGoodFramesWaiter, TimeWaiter,
    )

    if duration.unit == DurationUnit.GOOD_FRAMES:
        return PeriodGoodFramesWaiter(int(duration.value))
    if duration.unit == DurationUnit.EVENTS:
        return MEventsWaiter(float(duration.value) / 1_000_000.0)
    return TimeWaiter(seconds=float(duration.value))


@dataclass(frozen=True)
class DetectorGrouping:
    """One-based DAE spectrum numbers, not zero-based numpy array positions."""

    forward: tuple[int, ...]
    backward: tuple[int, ...]

    def __post_init__(self) -> None:
        f, b = set(self.forward), set(self.backward)
        if not f or not b or min(f | b) < 1 or len(f) != len(self.forward) or len(b) != len(self.backward):
            raise ValueError("Detector groups must be nonempty, positive and without duplicates")
        if f & b:
            raise ValueError("Forward and backward groups must not overlap")


def musr_longitudinal_grouping() -> DetectorGrouping:
    """MUSR manual (29 June 2011), section 2.6.1: 33–64 F, 1–32 B.

    This is not the ARGUS detector mapping and must be confirmed against the
    currently loaded MUSR DAE spectra table before a live experiment.
    """
    return DetectorGrouping(tuple(range(1, 33)), tuple(range(33, 65)))

def musr_transverse_grouping() -> DetectorGrouping:
    """
    Transverse grouping
    """
    return DetectorGrouping(tuple(range(9, 17)) + tuple(range(57, 65)), tuple(range(25,33)) + tuple(range(41, 49)))

def emu_grouping() -> DetectorGrouping:
    """EMU manual

    must be confirmed against the
    currently loaded EMU DAE spectra table before a live experiment.
    """
    return DetectorGrouping(tuple(range(1, 49)), tuple(range(49, 97)))

def hifi_grouping() -> DetectorGrouping:
    """HiFi manual 

    must be confirmed against the
    currently loaded HiFi DAE spectra table before a live experiment.
    """
    return DetectorGrouping(tuple(range(1, 33)), tuple(range(33, 65)))

class LiveSpectrumSnapshotter:
    """Snapshot DAE spectra in memory, before the run is ended and saved.

    A single `trigger_and_get_specdata` reads all selected spectra for the
    current period, excluding the DAE's junk time channel 0. Optionally read
    the TCB bin centres via a connected DaeSpectra subdevice. No NeXus file or
    Mantid workspace is loaded.

    `spectra` can be a subset for a general experiment. Muon forward/backward
    banks are an optional grouping, used only when those metrics are wanted.
    """

    def __init__(
        self,
        spectra: Sequence[int],
        *,
        grouping: DetectorGrouping | None = None,
        include_spectrum_matrix: bool = True,
        read_time_axis: bool = True,
        time_axis_unit: str = "ns",
    ) -> None:
        numbers = tuple(int(spec) for spec in spectra)
        if not numbers or len(set(numbers)) != len(numbers) or min(numbers) <= 0:
            raise ValueError("spectra must be nonempty, distinct positive spectrum numbers")
        if grouping is not None and not (set(grouping.forward) | set(grouping.backward)).issubset(numbers):
            raise ValueError("The detector grouping must be contained in the spectra selection")
        self.spectra = numbers
        self.grouping = grouping
        self.include_spectrum_matrix = include_spectrum_matrix
        self.read_time_axis = read_time_axis
        self.time_axis_unit = time_axis_unit

    async def capture(
        self,
        dae: Any,
        *,
        wall_seconds: float,
        run_number: int | None = None,
    ) -> dict[str, Any]:
        """Capture and validate a paused, live DAE run.

        No metric is evaluated here: callbacks can derive whatever objective
        they want from the acquisition record.
        """
        # `trigger_and_get_specdata` calls the DAE's UPDATE command and reads
        # the current in-memory spectra-data map, not the saved NeXus file.
        raw = await dae.trigger_and_get_specdata(np.asarray(self.spectra, dtype=np.int64))
        matrix = np.array(raw, dtype=np.float64, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.spectra) or matrix.shape[1] < 1:
            raise ValueError(f"Unexpected DAE spectrum matrix shape: {matrix.shape}")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("Invalid DAE spectrum counts")

        frames = int(await dae.period.good_frames.get_value())
        if frames <= 0:
            raise ValueError("DAE returned zero good frames; cannot normalise rate")

        events_millions = float(await dae.m_events.get_value())
        run_duration_seconds = float(await dae.period.run_duration.get_value())
        if run_number is None:
            run_number = int(await dae.controller.run_number.get_value())
        if run_number < 1:
            raise ValueError("Invalid live DAE run number")

        data: dict[str, Any] = {
            "run_number": run_number,
            "spectrum_numbers": np.asarray(self.spectra, dtype=np.int64),
            "total_counts": float(matrix.sum()),
            "good_frames": frames,
            "events_millions": events_millions,
            "wall_seconds": float(wall_seconds),
            "dae_run_duration_seconds": run_duration_seconds,
        }
        if self.include_spectrum_matrix:
            data["counts_by_spectrum"] = matrix

        if self.grouping is not None:
            ix = {s: i for i, s in enumerate(self.spectra)}
            f = matrix[[ix[s] for s in self.grouping.forward]].sum(axis=0)
            b = matrix[[ix[s] for s in self.grouping.backward]].sum(axis=0)
            data["forward_counts"] = f
            data["backward_counts"] = b

        if self.read_time_axis:
            # This DaeSpectra child is connected by Ophyd along with `dae`.
            tof = np.asarray(await dae.live_time_axis.read_tof(), dtype=float).copy()
            if tof.ndim != 1 or tof.size != matrix.shape[1]:
                raise ValueError(
                    f"DAE time bins ({tof.size}) do not match counts ({matrix.shape[1]})"
                )
            data["time_bins"] = tof
            data["time_bin_unit"] = self.time_axis_unit

        return data


async def acquire_snapshot_before_save(
    dae: Any,
    snapshotter: LiveSpectrumSnapshotter,
    pause_and_confirm: Callable[[Any], Awaitable[None]],
    *,
    max_wait_seconds: float | None,
    benchmark_recorder: Any | None = None,
) -> dict[str, Any]:
    """Perform a run and END/save after a successful BEGIN handshake.

    No RE nesting: called by the `AsyncStatus` implementation of the device's
    trigger, which is in turn invoked by Bluesky via `bp.count`.

    A failure during acquisition or capture is not hidden by a secondary END
    failure; both are surfaced via exception chaining when possible.
    """
    started = False
    error: BaseException | None = None
    snapshot: dict[str, Any] | None = None
    try:
        # Only END a run that this controller successfully began. If BEGIN
        # itself fails, ownership of the DAE run is ambiguous and it is safer
        # to require operator recovery than risk ENDing another user's run.
        await dae.controller.start_counting(dae)
        started = True
        wait_started = monotonic()
        if max_wait_seconds is None:
            await dae.waiter.wait(dae)
        else:
            await asyncio.wait_for(dae.waiter.wait(dae), timeout=max_wait_seconds)
        exposure_wall_seconds = monotonic() - wait_started
        if benchmark_recorder is not None:
            benchmark_recorder.record("dae_acquisition_wait", exposure_wall_seconds)
        phase_started = monotonic()
        await pause_and_confirm(dae)
        if benchmark_recorder is not None:
            benchmark_recorder.record("dae_pause", monotonic() - phase_started)
        phase_started = monotonic()
        snapshot = await snapshotter.capture(dae, wall_seconds=exposure_wall_seconds)
        if benchmark_recorder is not None:
            benchmark_recorder.record("dae_snapshot", monotonic() - phase_started)
    except BaseException as exc:
        error = exc
    finally:
        if started:
            phase_started = monotonic()
            try:
                # RunPerPointController(save_run=True) invokes END, not ABORT.
                await dae.controller.stop_counting(dae)
            except BaseException as end_exc:
                if error is None:
                    raise
                # Preserve the initial error and explain save failure too.
                if hasattr(error, "add_note"):
                    error.add_note(f"DAE END/save also failed: {end_exc!r}")
            finally:
                if benchmark_recorder is not None:
                    benchmark_recorder.record("dae_end_save", monotonic() - phase_started)
    if error is not None:
        raise error
    assert snapshot is not None
    return snapshot


async def acquire_period_snapshot(
    dae: Any,
    snapshotter: LiveSpectrumSnapshotter,
    *,
    max_wait_seconds: float | None,
    benchmark_recorder: Any | None = None,
) -> dict[str, Any]:
    """Count one period, pause it and capture its live data.

    The controller owns the surrounding run: ``stage`` begins it and
    ``unstage`` ends or aborts it. This helper therefore must not call
    ``stop_counting`` after the snapshot except when recovering from an
    acquisition failure.
    """
    started = False
    paused = False
    error: BaseException | None = None
    snapshot: dict[str, Any] | None = None
    try:
        await dae.controller.start_counting(dae)
        started = True
        wait_started = monotonic()
        if max_wait_seconds is None:
            await dae.waiter.wait(dae)
        else:
            await asyncio.wait_for(dae.waiter.wait(dae), timeout=max_wait_seconds)
        exposure_wall_seconds = monotonic() - wait_started
        if benchmark_recorder is not None:
            benchmark_recorder.record("dae_acquisition_wait", exposure_wall_seconds)
        phase_started = monotonic()
        await dae.controller.stop_counting(dae)
        if benchmark_recorder is not None:
            benchmark_recorder.record("dae_pause", monotonic() - phase_started)
        paused = True
        run_number = int(await dae.current_or_next_run_number.get_value())
        phase_started = monotonic()
        snapshot = await snapshotter.capture(
            dae,
            wall_seconds=exposure_wall_seconds,
            run_number=run_number,
        )
        if benchmark_recorder is not None:
            benchmark_recorder.record("dae_snapshot", monotonic() - phase_started)
    except BaseException as exc:
        error = exc
    finally:
        if started and not paused:
            try:
                await dae.controller.stop_counting(dae)
            except BaseException as pause_exc:
                if error is None:
                    raise
                if hasattr(error, "add_note"):
                    error.add_note(f"DAE pause during recovery also failed: {pause_exc!r}")
    if error is not None:
        raise error
    assert snapshot is not None
    return snapshot


def make_live_run_per_point_dae(
    *,
    duration: AcquisitionDuration,
    spectra: Sequence[int],
    grouping: DetectorGrouping | None = None,
    prefix: str | None = None,
    name: str = "optimisation_dae",
    max_wait_seconds: float | None = 600.0,
    include_spectrum_matrix: bool = True,
    read_time_axis: bool = True,
    time_axis_unit: str = "ns",
    benchmark_recorder: Any | None = None,
):
    """Create a RunPerPointController(save_run=True) DAE for live metrics.

    This factory is intended to run in the instrument's installed Bluesky
    environment. The returned device should be connected with
    `ophyd_async.plan_stubs.ensure_connected` before use.

    The DAE must be in SETUP with a valid spectra table/TCB configuration.
    Does not change any instrument hardware settings, interlocks or period
    configuration. Reads live spectrum map *before* END/save.
    """
    from ibex_bluesky_core.devices.dae import DaeSpectra, RunstateEnum
    from ibex_bluesky_core.devices.simpledae import (
        Reducer, RunPerPointController, SimpleDae,
    )
    from ibex_bluesky_core.utils import get_pv_prefix
    from ophyd_async.core import AsyncStatus, soft_signal_r_and_setter, wait_for_value

    if prefix is None:
        prefix = get_pv_prefix()
    if max_wait_seconds is not None and (not np.isfinite(max_wait_seconds) or max_wait_seconds <= 0):
        raise ValueError("max_wait_seconds must be positive or None")

    snapshotter = LiveSpectrumSnapshotter(
        spectra=spectra,
        grouping=grouping,
        include_spectrum_matrix=include_spectrum_matrix,
        read_time_axis=read_time_axis,
        time_axis_unit=time_axis_unit,
    )

    class LiveRunPerPointDae(SimpleDae):
        """SimpleDae variant with a live PRE-END snapshot (rather than POST-END)."""

        def __init__(self) -> None:
            self.last_snapshot: dict[str, Any] | None = None
            if read_time_axis:
                # `DaeSpectra` requires the full DAE prefix, e.g. IN:MUSR:DAE:
                self.live_time_axis = DaeSpectra(
                    f"{prefix}DAE:", spectra=snapshotter.spectra[0], period=1
                )
            # Scalar signals are emitted as Bluesky Event documents. The full
            # detector matrix is deliberately retained only in last_snapshot.
            self.snapshot_counts, self._counts_setter = soft_signal_r_and_setter(float, 0.0)
            self.snapshot_frames, self._frames_setter = soft_signal_r_and_setter(int, 0)
            self.snapshot_run_number, self._number_setter = soft_signal_r_and_setter(int, 0)
            super().__init__(
                prefix=prefix,
                name=name,
                controller=RunPerPointController(save_run=True),
                waiter=make_duration_waiter(duration),
                reducer=Reducer(),  # live spectrum reduction handled pre-END
            )
            self.add_readables(devices=[
                self.snapshot_counts, self.snapshot_frames, self.snapshot_run_number,
            ])

        @AsyncStatus.wrap
        async def trigger(self) -> None:
            self.last_snapshot = None
            # Do not disturb an externally-running DAE, and do not silently
            # acquire from a stale non-one period configuration.
            state = await self.run_state.get_value()
            if state != RunstateEnum.SETUP:
                raise RuntimeError(f"DAE must be in SETUP before optimisation acquisition, got {state}")
            n_periods = int(await self.number_of_periods.signal.get_value())
            if n_periods != 1:
                raise RuntimeError(f"Run-per-point requires exactly one DAE period; got {n_periods}")
            n_spectra = int(await self.num_spectra.get_value())
            if max(snapshotter.spectra) > n_spectra:
                raise RuntimeError(
                    f"Configured spectrum {max(snapshotter.spectra)} exceeds DAE count {n_spectra}"
                )

            async def pause_and_confirm(dae: Any) -> None:
                await dae.controls.pause_run.trigger(timeout=None)
                await wait_for_value(dae.run_state, RunstateEnum.PAUSED, timeout=20)

            snapshot = await acquire_snapshot_before_save(
                self, snapshotter, pause_and_confirm,
                max_wait_seconds=max_wait_seconds,
                benchmark_recorder=benchmark_recorder,
            )
            self.last_snapshot = snapshot
            self._counts_setter(float(snapshot["total_counts"]))
            self._frames_setter(int(snapshot["good_frames"]))
            self._number_setter(int(snapshot["run_number"]))

    return LiveRunPerPointDae()

def make_live_period_per_point_dae(
    *,
    duration: AcquisitionDuration,
    spectra: Sequence[int],
    grouping: DetectorGrouping | None = None,
    prefix: str | None = None,
    name: str = "optimisation_dae",
    max_wait_seconds: float | None = 600.0,
    include_spectrum_matrix: bool = True,
    read_time_axis: bool = True,
    time_axis_unit: str = "ns",
    benchmark_recorder: Any | None = None,
):
    """Create a PeriodPerPointController(save_run=True) DAE for live metrics.

    This factory is intended to run in the instrument's installed Bluesky
    environment. The returned device should be connected with
    `ophyd_async.plan_stubs.ensure_connected` before use.

    The DAE must be in SETUP with a valid spectra table/TCB configuration.
    Each staged device opens one saved DAE run, and each trigger acquires one
    period. Does not change any instrument hardware settings, interlocks or
    period configuration. Reads live spectrum map before the period is paused.
    """
    from ibex_bluesky_core.devices.dae import DaeSpectra
    from ibex_bluesky_core.devices.simpledae import (
        Reducer, PeriodPerPointController, SimpleDae,
    )
    from ibex_bluesky_core.utils import get_pv_prefix
    from ophyd_async.core import AsyncStatus, soft_signal_r_and_setter

    if prefix is None:
        prefix = get_pv_prefix()
    if max_wait_seconds is not None and (not np.isfinite(max_wait_seconds) or max_wait_seconds <= 0):
        raise ValueError("max_wait_seconds must be positive or None")

    snapshotter = LiveSpectrumSnapshotter(
        spectra=spectra,
        grouping=grouping,
        include_spectrum_matrix=include_spectrum_matrix,
        read_time_axis=read_time_axis,
        time_axis_unit=time_axis_unit,
    )

    class LivePeriodPerPointDae(SimpleDae):
        """SimpleDae variant with a live PRE-END snapshot (rather than POST-END)."""

        def __init__(self) -> None:
            self.last_snapshot: dict[str, Any] | None = None
            if read_time_axis:
                # `DaeSpectra` requires the full DAE prefix, e.g. IN:MUSR:DAE:
                self.live_time_axis = DaeSpectra(
                    f"{prefix}DAE:", spectra=snapshotter.spectra[0], period=1
                )
            # Scalar signals are emitted as Bluesky Event documents. The full
            # detector matrix is deliberately retained only in last_snapshot.
            self.snapshot_counts, self._counts_setter = soft_signal_r_and_setter(float, 0.0)
            self.snapshot_frames, self._frames_setter = soft_signal_r_and_setter(int, 0)
            self.snapshot_run_number, self._number_setter = soft_signal_r_and_setter(int, 0)
            super().__init__(
                prefix=prefix,
                name=name,
                controller=PeriodPerPointController(save_run=True),
                waiter=make_duration_waiter(duration),
                reducer=Reducer(),  # live spectrum reduction handled pre-END
            )
            self.add_readables(devices=[
                self.snapshot_counts, self.snapshot_frames, self.snapshot_run_number,
            ])

        @AsyncStatus.wrap
        async def trigger(self) -> None:
            self.last_snapshot = None
            n_spectra = int(await self.num_spectra.get_value())
            if max(snapshotter.spectra) > n_spectra:
                raise RuntimeError(
                    f"Configured spectrum {max(snapshotter.spectra)} exceeds DAE count {n_spectra}"
                )

            snapshot = await acquire_period_snapshot(
                self,
                snapshotter,
                max_wait_seconds=max_wait_seconds,
                benchmark_recorder=benchmark_recorder,
            )
            self.last_snapshot = snapshot
            self._counts_setter(float(snapshot["total_counts"]))
            self._frames_setter(int(snapshot["good_frames"]))
            self._number_setter(int(snapshot["run_number"]))

    return LivePeriodPerPointDae()

def make_musr_live_dae(
    *,
    duration: AcquisitionDuration,
    grouping: DetectorGrouping | None = None,
    spectra: Sequence[int] | None = None,
    **kwargs: Any,
):
    """MUSR longitudinal-mode convenience factory, *not* an ARGUS config.

    Explicitly verify detector numbering against the CURRENT spectra table.
    For transverse mode, provide the correct grouping, or use the generic
    factory. The 2011 manual groups are not valid for every configuration.
    """
    if grouping is None:
        grouping = musr_longitudinal_grouping()
    if spectra is None:
        spectra = tuple(range(1, 65))
    return make_live_run_per_point_dae(
        duration=duration, spectra=spectra, grouping=grouping, **kwargs,
    )

def make_musr_live_dae_ppp(
    *,
    duration: AcquisitionDuration,
    grouping: DetectorGrouping | None = None,
    spectra: Sequence[int] | None = None,
    **kwargs: Any,
):
    """MUSR longitudinal-mode convenience factory.

    Controls creation of a PeriodPerPointController(save_run=True) DAE for live metrics, with a snapshot
    taken before the run is ended and saved.
    """
    if grouping is None:
        grouping = musr_longitudinal_grouping()
    if spectra is None:
        spectra = tuple(range(1, 65))
    return make_live_period_per_point_dae(
        duration=duration, spectra=spectra, grouping=grouping, **kwargs,
    )

def make_live_dae(
    *,
    duration: AcquisitionDuration,
    instrument: str,
    period_per_point: bool = False,
    **kwargs: Any,
):
    """Create a live DAE for optimisation, either run-per-point or period-per-point.

    The default is run-per-point, which is slower but more robust to DAE
    configuration changes. Period-per-point is faster but requires a stable
    DAE configuration and a single period per run.
    """
    if instrument.lower() == "musr_longitudinal":
        grouping = musr_longitudinal_grouping()
        spectra = tuple(range(1, 65))
    elif instrument.lower() == "musr_transverse":
        grouping= musr_transverse_grouping()
        spectra = tuple(range(1, 65))
    elif instrument.lower() == "emu":
        grouping = emu_grouping()
        spectra = tuple(range(1, 97))
    elif instrument.lower() == "hifi":
        grouping = hifi_grouping()
        spectra = tuple(range(1, 65))
    else:
        raise ValueError(f"Unknown instrument {instrument!r}; must be one of 'musr_longitudinal', 'emu', or 'hifi'")
    
    if period_per_point:
        return make_live_period_per_point_dae(
            duration=duration, spectra=spectra, grouping=grouping, **kwargs,
        )
    else:
        return make_live_run_per_point_dae(
            duration=duration, spectra=spectra, grouping=grouping, **kwargs,
        )