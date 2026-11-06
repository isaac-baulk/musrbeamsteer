# ISIS live DAE acquisition for N-dimensional Bluesky minimisation

This module uses the **current** ISIS `ibex_bluesky_core` Bluesky DAEs and
**never reads the NeXus file to evaluate the optimisation metric**.

The design is suitable for MUSR after verifying its current detector mapping,
EPICS connectivity and device limits, and can be applied to other instruments by
providing different spectra/grouping and positioners. The steering example
previously provided in this discussion was for **ARGUS**, not MUSR.

## Documented interfaces

- ISIS IBEX is the EPICS-based control system; the linked top-level
  [IBEX repository](https://github.com/ISISComputingGroup/IBEX) tracks the
  project, while the relevant Python implementation is
  [`ibex_bluesky_core`](https://github.com/ISISComputingGroup/ibex_bluesky_core).
- [ISIS SimpleDae documentation](https://isiscomputinggroup.github.io/ibex_bluesky_core/devices/dae.html)
  describes `RunPerPointController`, `PeriodGoodFramesWaiter`,
  `MEventsWaiter` and `TimeWaiter`.
- The published [SimpleDae implementation](https://isiscomputinggroup.github.io/ibex_bluesky_core/_modules/ibex_bluesky_core/devices/simpledae.html)
  shows that the default trigger performs BEGIN, WAIT, END and only then REDUCE.
- The [DAE spectrum map](https://isiscomputinggroup.github.io/ibex_bluesky_core/_modules/ibex_bluesky_core/devices/dae.html)
  is exposed by `trigger_and_get_specdata()` and refreshed with `UPDATE`.
- [DAE PV documentation](https://isiscomputinggroup.github.io/ibex_developers_manual/specific_iocs/dae/DAE-PVs.html)
  explains that spectrum `YC` gives counts per bin, not counts per unit time.

## Acquisition lifecycle

`make_live_run_per_point_dae` creates a subclass of `SimpleDae` with
`RunPerPointController(save_run=True)`. Each call to `bp.count([dae], num=1)`:

1. Confirms the DAE is in `SETUP`, configured with exactly **one period**, and
   has enough spectra to serve the requested detectors.
2. Starts a new DAE run; controller records the actual ISIS DAE run number.
3. Waits for the selected threshold **while the run is active**.
4. Pauses the DAE and confirms that its state is `PAUSED`.
5. Reads in-memory spectrum data from the current DAE with
   `trigger_and_get_specdata`; also reads frame count, event count, run time,
   and (optionally) the time axis. The snapshot is an immutable *copy of the
   arrays at this point in time*, not a view into the DAE buffers.
6. Issues `END` via the standard run-per-point controller, **saving the run**.
   This is done even if waiting or snapshotting fails, where feasible.
7. Emits lightweight scalar Bluesky Event fields and returns an acquisition
   record containing the snapshot, actual DAE run number and Bluesky UID.
8. Computes the optimisation metric using the in-memory record. The next
   measurement is only started when the previous `END` finishes.

**Performance:** This removes the wait for NeXus file availability and Mantid
file reading/processing in the feedback loop. It **does not remove time spent
in DAE UPDATE, END or saving**. Run-per-point mode cannot begin the next run
while a previous run is still ending, and the actual feedback time has to be
measured on the instrument. For minimum end-to-end latency, a streamed data path may eventually be
necessary. `make_live_period_per_point_dae` is also available when multiple
points should share one saved DAE run: staging begins the run, each point
counts into the next DAE period, and unstaging ends/saves the run.

The acquisition adapter supports both controller lifecycles:

```python
from musrbeamsteer.acquisition import make_live_ibex_dae_acquire

# Run-per-point: bp.count stages and saves the DAE for each measurement.
acquire = make_live_ibex_dae_acquire(dae)

# Period-per-point: stage once around the complete minimize_plan.
acquire = make_live_ibex_dae_acquire(dae, shared_run=True)
yield from bps.stage(dae)
try:
    yield from minimize_plan(..., acquire=acquire)
finally:
    yield from bps.unstage(dae)
```

Do not use `shared_run=True` without the matching outer stage/unstage; the
period controller's run is owned by those operations.

Spectrum readback is a paused snapshot, not a continuous stream. The DAE
client issues `UPDATE`, reads `SPECDATA`, and selects the current
`period_num`; no IBEX data livestreamer or separate streaming configuration is
needed. The time axis is read from the TCB PVs and is shared by periods.

The period-scoped frame and run-duration signals are used for each point.
The existing `events_millions` field is the DAE-wide counter; do not interpret
it as a per-period event count when using a shared run.

The MUSR period-per-point example presets `DAE:NUMPERIODS` to 60 before
staging. It rejects `max_measurements > 60` before opening the run, preventing
the optimiser from requesting more periods than the configured memory-card
budget.

**Failure behaviour:** Once BEGIN completes successfully, the device attempts
END/save in a `finally` block even if waiting or DAE snapshotting fails. This
may save an incomplete run. If the initial BEGIN handshake itself fails, the
plan will not blindly END a run it may not own; check the DAE state manually. `max_wait_seconds` limits the wait
for frames/events/seconds, **not** the time spent in END/save. If the DAE fails
to complete END, manual instrument-side recovery may be necessary.

## Acquisition target

The configuration is declarative; only the waiter changes:

```python
from musrbeamsteer.isis_dae import AcquisitionDuration

AcquisitionDuration.good_frames(500)
AcquisitionDuration.events(1_000_000)   # converts to MEventsWaiter(1.0)
AcquisitionDuration.seconds(60.0)       # wall-clock time, including beam veto
```

`MEventsWaiter` takes *millions* of events, so `events(n)` divides `n` by
1,000,000. The `TimeWaiter` uses wall time irrespective of DAE beam state.
The other two waiters can time out using `max_wait_seconds` if beam stops.

## MUSR detector mapping

The attached **MuSR Manual (29 June 2011), section 2.6.1, page 12** specifies
64 scintillator detectors arranged in two rings. In longitudinal geometry,
forward = **33–64** and backward = **1–32**. This differs from the ARGUS
example with 192 numbered channels. The manual also lists alternative
transverse groups. The current loaded MUSR spectra table MUST be checked before
using these defaults on live hardware: the manual is old and does not document
the current IBEX configurations.

`make_musr_live_dae()` uses the historical longitudinal grouping as a
configurable default. For any other instrument or geometry use:

```python
from musrbeamsteer.isis_dae import (
    AcquisitionDuration, DetectorGrouping, make_live_run_per_point_dae,
)

dae = make_live_run_per_point_dae(
    duration=AcquisitionDuration.events(300_000),
    spectra=list(range(1, 65)),
    grouping=DetectorGrouping(
        forward=tuple(range(33, 65)), backward=tuple(range(1, 33)),
    ),
)
```

For plain count rate, `grouping=None` can be used on any configured detector
subset. Do not reuse the historical 64-detector default for EMU or ARGUS.

## Metrics and fields

The live snapshot is a `record['data']` dictionary. Important fields:

| Key | Meaning |
| --- | --- |
| `run_number` | ISIS DAE run number recorded on BEGIN |
| `good_frames` | Frames measured in this DAE period |
| `events_millions` | DAE's MEVENTS counter |
| `wall_seconds` | Actual wall-clock time spent waiting |
| `dae_run_duration_seconds` | DAE's reported whole-run duration |
| `total_counts` | Integrated counts of all selected spectra and all time bins |
| `counts_by_spectrum` | Optional `(N_spectra, N_bins)` numpy matrix |
| `spectrum_numbers` | One-based DAE spectrum numbers in corresponding row order |
| `forward_counts`, `backward_counts` | Optional arrays of time-bin counts (group sums) |
| `time_bins` | Optional DAE time-bin centres, usually ns on muon instruments |
| `time_bin_unit` | Configured time unit label, default `ns` |

Each point also has a fresh *Bluesky* `uid`, which is distinct from the numeric
*ISIS DAE run number*. No UID-to-run-number guess or NeXus reader is needed.

The independent metrics work directly on these fields:

```python
from musrbeamsteer.metrics import (
    count_rate_metric, initial_asymmetry_metric,
    asymmetry_squared_count_rate_metric,
)

rate = count_rate_metric()  # -(total_counts / good_frames)
initial = initial_asymmetry_metric(alpha=1.0, start_bin=40, bins=20)
figure = asymmetry_squared_count_rate_metric(alpha=1.0, start_bin=40, bins=20)
```

The bin values above are *illustrative*, not a calibrated MUSR `t0`.
Alternatively use `time_window=(start, stop)` in the raw DAE time-axis units.
The early-time asymmetry is NOT a fitted `A_0`. Apply experiment-specific time
zero, background, detector efficiency/alpha and dead-time corrections before
interpreting it physically. `A² * count rate` combines the selected early-time
asymmetry with the **full selected-spectrum rate**, not only early-time counts.
The count-rate metric can use `per='second'`, but this is an uncorrected
wall-clock rate, not the beam-veto-adjusted frame rate.

Independent metrics are `functools.lru_cache`-cached by acquisition UID.
The new `musrbeamsteer.musr_metrics` module provides
`integrated_asymmetry_metric(...)` (per-record, cached) and
`pca_asymmetry_metric(...)` (whole-dataset, uncached), both from live
forward/backward histograms and `time_bins` with no NeXus files. For example:

```python
from musrbeamsteer.musr_metrics import (
    integrated_asymmetry_metric, pca_asymmetry_metric,
)
metric = integrated_asymmetry_metric(alpha=CALIBRATED_ALPHA,
                                     time_window_ns=(POST_T0_NS, 8000))
# or metric = pca_asymmetry_metric(...)
```

This is a direct-histogram PCA, not the old TT_PCA/Mantid preprocessing.
Legacy file-based PCA callbacks still use the old, non-live path.

## Using the minimiser

See [`examples/musr_run_per_point.py`](examples/musr_run_per_point.py) for a
plan factory taking arbitrary instrument positioners, safe bounds, a duration,
and a metric. Run it in the existing instrument RunEngine only **after**
checking that the DAE is in SETUP, all positioners and PVs are connected,
interlocks and motion limits are appropriate, the experiment labels are set,
and an instrument scientist has approved the selected axes and ranges. On
ISIS, `run_result = RE(plan)` returns a `RunEngineResult`, so use
`run_result.plan_result` to access the optimiser `SearchResult`.

The old `examples/bluesky_ibex_example.py` describes an **ARGUS-style** layout,
not MUSR. It is only retained as a historical example.

## Verification and remaining site checks

The local tests mock the DAE/Ophyd/Bluesky calls. They check waiters,
metric arithmetic, PRE-END snapshot ordering, error/timeout cleanup, detector
grouping and generic GP/KG behaviour. They **do not** demonstrate a real ISIS
hardware connection or a completed run on MUSR. Before beamline use:

- Confirm the actual `ibex_bluesky_core`, `bluesky`, `ophyd-async` versions.
- Confirm a real DAE is in SETUP, has 1 software period, and gives the
  expected spectrum shape and time-bin units.
- Confirm forward/backward detector groups and experiment-specific `t0`,
  `alpha`, corrections and dead-time behaviour.
- Check `trigger_and_get_specdata` speed with a short real run and whether
  END blocks for NeXus finalisation. Inspect the saved file's run number.
- Start with a DAE-only run (no motor movement), then a limited, approved
  single-axis test before using multi-dimensional random search.
- Constrain all optimisation/random-exploration coordinates to instrument-safe
  limits and consider the possibility of jointly unsafe settings even when
  individual coordinates are within bounds.

The `trigger_and_get_specdata` API copies the **entire** DAE spectrum-data
map before selecting the requested detector rows. Confirm adequate EPICS array
sizes and the instrument's spectrum/period/time-channel product before using
large time-bin tables. The acquisition path does not wait for the historical
`Run 0` NeXus file, which the 2011 MUSR manual warns can remain stale during
very short measurements.
