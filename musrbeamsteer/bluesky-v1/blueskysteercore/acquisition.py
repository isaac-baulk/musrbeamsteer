"""Bluesky acquisition adapters: one Bluesky run per measurement point.

No nested RunEngine, no Genie, no Mantid and no NeXus file read.
The ISIS live DAE device supplies a snapshot taken before END/save.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def make_bluesky_acquire(
    detectors,
    *,
    metadata: dict | None = None,
    transform: Callable[[dict], dict] | None = None,
    stage_detectors: bool = True,
):
    """Wrap a Bluesky acquisition plan returning a measurement record.

    `detectors` are already-connected Bluesky/Ophyd devices. One fresh UID
    identifies each measurement, including repeated coordinates. When
    ``stage_detectors`` is false, the caller owns staging and unstaging; this
    is required when a period-per-point DAE run is shared across measurements.
    """
    import bluesky.plans as bp
    import bluesky.plan_stubs as bps
    from bluesky.preprocessors import run_decorator, subs_wrapper

    def acquire(coordinate):
        events: list[dict] = []
        run_uids: list[str] = []

        def receiver(name, doc):
            if name == "start":
                run_uids.append(str(doc["uid"]))
            elif name == "event":
                events.append(dict(doc["data"]))

        plan_metadata = {
            **(metadata or {}),
            "optimisation_coordinate": list(map(float, coordinate)),
        }
        if stage_detectors:
            plan = bp.count(list(detectors), num=1, md=plan_metadata)
        else:
            @run_decorator(md=plan_metadata)
            def unstaged_count():
                yield from bps.one_shot(detectors)

            plan = unstaged_count()
        yield from subs_wrapper(plan, receiver)
        if not run_uids or not events:
            raise RuntimeError("Bluesky acquisition did not emit Start and Event documents")
        # A RunStart UID is a unique acquisition identifier; do not mistake the
        # plan's return value or the numeric ISIS run number for the same UID.
        uid = run_uids[-1]
        data = dict(events[-1])
        if transform is not None:
            data = dict(transform(data))
        return {"uid": uid, "coordinate": tuple(map(float, coordinate)), "data": data}

    return acquire


def make_ibex_dae_acquire(
    dae: Any,
    *,
    transform: Callable[[dict], dict] | None = None,
    run_number_from_uid: Callable[[str], int | str] | None = None,
    require_live_snapshot: bool = False,
    stage_per_acquisition: bool = True,
):
    """Acquire one ISIS DAE run/point, taking metrics from LIVE data when available.

    For the live run- or period-per-point DAE factories this includes the
    complete in-memory snapshot, read while the DAE was PAUSED and before
    END/save. For a plain `SimpleDae` the adapter only collects its emitted
    scalar Event data.

    For the live device, `record['run_number']` is the actual ISIS DAE number,
    obtained during acquisition, never inferred from UID.
    Set ``stage_per_acquisition=False`` when the caller stages the DAE once
    around multiple acquisitions, as with ``PeriodPerPointController``.
    The optional legacy mapping is retained only for old external adapters.
    """
    base = make_bluesky_acquire(
        [dae],
        transform=transform,
        stage_detectors=stage_per_acquisition,
    )

    def acquire(coordinate):
        record = yield from base(coordinate)
        snapshot = getattr(dae, "last_snapshot", None)
        if snapshot is None and require_live_snapshot:
            raise RuntimeError("Live DAE snapshot is missing; refusing to use stale Event data")
        if snapshot is not None:
            # A newly acquired live snapshot overrides optional Event scalar
            # aliases, while keeping all original Event fields for diagnostics.
            record["data"] = {**record["data"], **snapshot}
            record["run_number"] = int(snapshot["run_number"])
        elif run_number_from_uid is not None:
            record["run_number"] = run_number_from_uid(record["uid"])
        return record

    return acquire


def make_live_ibex_dae_acquire(dae: Any, *, shared_run: bool = False):
    """Require an in-memory PRE-END DAE snapshot for every optimisation point.

    ``shared_run=False`` uses one saved DAE run per measurement. With
    ``shared_run=True``, the caller must stage and unstage the DAE around the
    complete acquisition sequence; each measurement is then one period in the
    shared saved run.
    """
    return make_ibex_dae_acquire(
        dae,
        require_live_snapshot=True,
        stage_per_acquisition=not shared_run,
    )


class _SyntheticSummaryReadable:
    """Minimal synchronous Bluesky Readable for scalar synthetic Event data.

    The complete time-binned measurement remains in the returned acquisition
    record. Only a few scalar diagnostics are emitted as Event documents, so
    a databroker doesn't receive an entire histogram for each optimisation
    point. No Ophyd/EPICS/DAE connection is required for this detector.
    """

    _labels = ("total_counts", "good_frames", "sample_fraction", "rate_fraction", "expected_counts_per_frame")

    def __init__(self, name: str = "synthetic_dae") -> None:
        from time import time
        self.name = name
        self._timestamp = time()
        self._values = {key: 0.0 for key in self._labels}

    def update(self, snapshot: dict) -> None:
        from time import time
        self._timestamp = time()
        self._values = {key: float(snapshot[key]) for key in self._labels}

    def describe(self) -> dict:
        return {
            f"{self.name}_{key}": {
                "source": f"SIM:{self.name}:{key}",
                "dtype": "number",
                "shape": [],
            }
            for key in self._labels
        }

    def read(self) -> dict:
        return {
            f"{self.name}_{key}": {"value": value, "timestamp": self._timestamp}
            for key, value in self._values.items()
        }


def make_synthetic_acquire(
    simulator: Callable[[Any], dict],
    *,
    delay_seconds: float = 0.0,
    metadata: dict | None = None,
):
    """Return a *Bluesky plan* that replaces the DAE acquisition backend.

    ``simulator`` is a callable taking an N-D coordinate and returning a
    DAE-shaped dictionary, e.g. ``functools.partial(generate_synthetic_musr_data,
    center=..., curvature=..., delta_us_inv=...)``.

    Each call opens a Bluesky run with a real RunStart UID, optionally yields
    a Bluesky sleep, emits one Event of scalar diagnostics and closes the run.
    Arrays live in record['data'], identical to the real DAE metric contract.
    The synthetic path never contacts the DAE and never saves an ISIS run.

    A newly generated measurement is assigned a *fresh* UID even if the
    coordinate was measured before, so independent metric caches do not
    eliminate the stochastic repeats used to estimate observation noise.
    """
    import math
    import numpy as np

    if not callable(simulator):
        raise TypeError("simulator must be a callable")
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay_seconds must be finite and >= 0")
    summary = _SyntheticSummaryReadable()

    def acquire(coordinate):
        from bluesky import plan_stubs as bps

        x = np.asarray(coordinate, dtype=float)
        if x.ndim != 1 or not np.all(np.isfinite(x)):
            raise ValueError("synthetic coordinate must be a finite 1-D vector")
        run_uid = yield from bps.open_run(
            md={**(metadata or {}), "optimisation_coordinate": x.tolist(), "synthetic": True},
        )
        try:
            if delay_seconds:
                yield from bps.sleep(delay_seconds)
            snapshot = simulator(x.copy())
            if not isinstance(snapshot, dict):
                raise TypeError("simulator must return a DAE-shaped dict")
            summary.update(snapshot)
            yield from bps.create(name="primary")
            yield from bps.read(summary)
            yield from bps.save()
        except Exception as exc:
            yield from bps.close_run(exit_status="fail", reason=str(exc))
            raise
        else:
            yield from bps.close_run()
        return {
            "uid": str(run_uid),
            "coordinate": tuple(map(float, x)),
            "data": snapshot,
            "synthetic": True,
        }

    return acquire
