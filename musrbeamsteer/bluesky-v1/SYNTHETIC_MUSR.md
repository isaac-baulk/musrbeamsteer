# Beam-off MUSR Bluesky steering test: synthetic DAE

The synthetic acquisition backend moves the **real two MUSR steering
magnets** through Bluesky but obtains the muon data from a Python simulator.
It does not connect to the DAE, perform a DAE run, or write NeXus files.
The live backend still performs **one saved DAE run per point** and obtains
its histogram directly from the paused DAE, without reading a NeXus file.

**The previously supplied Danfysik example was from ARGUS, not MUSR.** The
setpoint/readback PV names, safe current ranges and tolerances for MUSR must
be provided from the actual instrument configuration. Never substitute ARGUS
PV names or assume the settings in this repository are hardware-safe.

## Quick start on MUSR

The example [`examples/musr_steering_synthetic.py`](examples/musr_steering_synthetic.py)
now contains the values supplied for the MUSR steering test:

```python
USE_SYNTHETIC = True
STEER_1_SETPOINT_PV = "IN:MUSR:KEPCO_04:CURRENT:SP"
STEER_1_READBACK_PV = "IN:MUSR:KEPCO_04:CURRENT"
STEER_2_SETPOINT_PV = "IN:MUSR:KEPCO_05:CURRENT:SP"
STEER_2_READBACK_PV = "IN:MUSR:KEPCO_05:CURRENT"

SAFE_BOUNDS = ((-0.5, 0.5), (-1.5, 1.5))
READBACK_TOLERANCES = (0.01, 0.01)
CURRENT_PRECISIONS = [3, 3]
SETTLE_SECONDS = 1.0
MOVE_TIMEOUT_SECONDS = 30.0

DURATION = AcquisitionDuration.events(1_000_000)
CENTER = [0.1, -0.3]    # TRUE synthetic centre (unknown to the optimiser)
CURVATURE = [3.0, 3.0]
METRIC_NAME = "integrated_asymmetry"   # or "pca"
METRIC_TIME_WINDOW_NS = (0.0, 30000.0)
METRIC_REBIN = 1  # native 16 ns bins (8 would be 128 ns)
MAX_MEASUREMENTS = 50
INITIAL_REPEATS = 2
VERBOSE = 2
MINIMUM_ESTIMATE_POINTS = 1024
```

**These are operator-provided test settings, not independently validated hardware
limits.** Confirm both actual readbacks are within `SAFE_BOUNDS` and that the
full rectangle is approved for the two magnets before executing. Synthetic
mode never starts the DAE, but still writes real steering setpoints.

With the existing `mantid_env` and project installation, from **CMD**:

```cmd
cd /d "C:\Users\musr\Documents\Isaac\musr_beamsteer_bluesky"
set "PATH=%CD%;%PATH%"
python -c "from bluesky import RunEngine; from examples.musr_steering_synthetic import make_plan; RE = RunEngine({}); RE(make_plan())"
```

`set "PATH=%CD%;%PATH%"` is relevant only when a compatible `caRepeater.exe`
(and all its necessary EPICS DLLs) is already in that directory. It does not
fix missing DLLs or gateway permissions.

### Gaussian initial design around the operator's guess

`INITIAL_GUESS = None` (the default) **prompts every time** before connecting
to hardware:

```text
Initial guess for 2 steering currents within ((-0.5, 0.5), (-1.5, 1.5)), e.g. [0.0, -0.2]:
```

Enter e.g. `[0.0, -0.2]`. The parser also accepts `0.0, -0.2`. The guess
must be inside the bounds with at least the specified margin; invalid guesses
are rejected and re-prompted. To skip prompting in automation, set:

```python
INITIAL_GUESS = [0.0, -0.2]
```

The guess is the *mean* of the Gaussian, not the synthetic truth
(`CENTER`), and **it is not necessarily one of the acquired points**.
For dimension i:

```
sigma_i = min(INITIAL_GUESS[i] - SAFE_BOUNDS[i][0],
              SAFE_BOUNDS[i][1] - INITIAL_GUESS[i]) / 3
```

`gaussian_initial_coordinates` samples **1 + 2N distinct N-dimensional
points** by default (5 for the 2-magnet example). The first point is acquired
twice for estimating noise, giving 6 initial measurements in 2D. Samples
outside the strict interior are **rejected, not clipped**. The default
`INITIAL_EDGE_MARGIN_FRACTION = 0.02` excludes 2% of the entire width on
each side; precision rounding is checked against this margin too.
`RANDOM_SEED` makes the design repeatable. `INITIAL_DESIGN_POINTS` can override
the number of Gaussian initial positions.

**Only initial sampling uses this Gaussian and 2% margin.** Subsequent
Knowledge-Gradient proposals and warning-triggered random exploration can
reach either boundary within `SAFE_BOUNDS`, so those bounds must themselves
be safe. Use narrower hardware-approved bounds if the edge cannot be visited.

The `make_simulator()` factory freezes the synthetic settings using
`functools.partial(generate_synthetic_musr_data, center=..., curvature=...,
bounds=..., delta_us_inv=..., rng=...)`; a fresh RNG per optimisation run
returns new Poisson noise at every repeat.

## Synthetic time-bin definition

There are **2000 bins of width 16 ns**, with *edges* at
`0, 16, 32, ..., 32000 ns`. The last bin is `[31984, 32000) ns`. The
`time_bins` array contains **centres** `8, 24, ..., 31992 ns`, consistent
with the in-memory live DAE metric interface. `time_bin_edges` also appears
in the synthetic record, for consumers requiring exact edges.

The mean number of counts per time bin is proportional to

\[
N_t \propto e^{-t/\tau_\mu},\quad \tau_\mu=2.1969811\ \mu s,
\]

normalised so that the expected total counts over the 32 us time window
is `good_frames * expected_counts_per_frame`. Each detector/bin is an
**independent Poisson count**. Hence `Var(N)=E[N]` and
`std(N)=sqrt(E[N])`; the *expected* standard deviation decays as
`exp(-t/(2*tau_mu))`. The record includes `count_std_by_spectrum =
sqrt(observed counts)` as the usual estimate.

## Spatial model: KT mixture and independent count-rate profile

For N-dimensional coordinate \(\mathbf{x}\), centre \(\mathbf{c}\), and
curvatures \(k_i\), the sample fraction is

\[
w(\mathbf{x}) = \exp\left(-\sum_i k_i(x_i-c_i)^2\right).
\]

`CURVATURE` affects *how quickly the KT shape is replaced by flat
background*, but not the total number of muons per frame.

The static Gaussian zero-field Kubo–Toyabe function is

\[
G_{KT}(t) = \frac{1}{3} + \frac{2}{3}(1-\Delta^2 t^2)
                      \exp(-\Delta^2 t^2/2).
\]

The asymmetry-producing polarisation is

\[
P(t,\mathbf{x}) = w(\mathbf{x})G_{KT}(t) + [1-w(\mathbf{x})] \cdot 1,
\]

so at the centre the response is pure KT and far away it is flat in
time. **The underlying initial asymmetry is exactly \(A(0)=A_0\) everywhere**.
The late-time asymmetry and its integrated value vary with position.

To control the total count rate independently, for each dimension \(i\)
normalise the distance from centre to the **boundary in the direction of
travel**:

\[
d_i(\mathbf{x}) = \begin{cases}
 (x_i-c_i)/(b_{i,\max}-c_i), & x_i\ge c_i,\\
 (c_i-x_i)/(c_i-b_{i,\min}), & x_i<c_i.
\end{cases}
\]

Set \(r(\mathbf{x})=\max_i d_i(\mathbf{x})\), and

\[
R(\mathbf{x}) = R_0 \times (0.10)^{r(\mathbf{x})^2}.
\]

Consequently:

- At the centre: \(r=0\), expected muons/frame are `COUNTS_PER_GOOD_FRAME`.
- **At every boundary face and every corner**: \(r=1\), expected
  muons/frame are exactly **10%** of the centre.
- At interior positions: the rate is between 10% and 100%.
- The central coordinate must be *strictly inside* each parameter bound.

The Chebyshev-style radius was chosen to make **every** face—not only
selected axis endpoints or the corners—exactly 10%. It is a convenient,
configurable synthetic beam profile rather than a physical transport model.
For a different profile, replace `spatial_rate_fraction()` in `synthetic.py`.

For time bin \(t\), the expected forward/backward group counts are

\[
E[F_t] = N_t \frac{1+A_0P(t,\mathbf{x})}{2},\qquad
E[B_t] = N_t \frac{1-A_0P(t,\mathbf{x})}{2}.
\]

Each group is split evenly among the corresponding detectors. The simulator
uses the historical MUSR longitudinal grouping (forward 33–64 and backward
1–32) from section 2.6.1 of the supplied 2011 MUSR manual. This grouping
must be verified against the current DAE spectra table before real runs.

## Acquisition duration (same three options for live and synthetic)

```python
DURATION = AcquisitionDuration.good_frames(500)
# DURATION = AcquisitionDuration.events(1_000_000)
# DURATION = AcquisitionDuration.seconds(10.0)
```

In synthetic good-frame and seconds modes, frame exposure is independent of
position, so edge measurements collect ~10% as many counts as centre ones.
For `events(n)`, the simulator calculates the number of frames using the
**position-dependent expected count rate**; edge measurements therefore
require roughly 10 times as many frames for the same event target.
Poisson totals fluctuate about the event target: there is no real event
threshold detector in synthetic mode. Unless `delay_seconds` is set on
`make_synthetic_acquire`, simulation does not physically wait that long.

## Direct-histogram metrics: integrated asymmetry or PCA

The original code's integrated-asymmetry metric is an inverse-variance-
weighted average of the time-dependent asymmetry. That same definition is
implemented for live in-memory histograms:

```python
from musrbeamsteer.musr_metrics import (
    integrated_asymmetry_metric, pca_asymmetry_metric,
)

metric = integrated_asymmetry_metric(
    alpha=1.0, time_window_ns=(0, 8000), rebin=8,
)
# Or (no change to acquisition/Bluesky plan):
metric = pca_asymmetry_metric(
    alpha=1.0, time_window_ns=(0, 8000), rebin=8,
)
```

Both work directly from `forward_counts`, `backward_counts` and `time_bins`
without waiting for NeXus files. Both ignore empty/unreliable time bins;
`rebin=8` combines eight 16 ns channels into about 128 ns bins. The
**integrated metric is smaller near the KT centre**, so it is minimised
as-is (no sign inversion). It uses `functools.lru_cache` per unique
measurement UID, since old values do not change when new data are added.

The PCA metric builds a matrix of *asymmetry curves* (not total detector
counts), centres that matrix, then takes first-component scores. Because the
sign of an eigenvector is arbitrary, it orients PC1 to correlate positively
with the weighted integrated asymmetry, so the centre has a lower score in
a dominant-KT example. It is a **global metric**: every new measurement
recomputes scores for *all* observations, and caching is intentionally off.
This PCA operates directly on histograms. It is not the full, legacy
TT_PCA/Mantid preprocessing pipeline.

The true **initial** asymmetry remains constant even though count rate
decreases at edges; using initial-asymmetry alone would not reliably identify
the centre. Count-rate by itself now has an optimum at the centre, but the
example defaults to integrated asymmetry as requested.

**For real DAE measurements:** set the correct time-zero and post-t0 window,
check DAE time units, and calibrate alpha, deadtime and detector/background
corrections. These simple live reductions do not supply the full Mantid
calibration chain automatically.

## What the synthetic acquisition record contains

Each point has a unique Bluesky `uid`, `coordinate`, and a `data` dictionary
that shares these fields with the live DAE snapshot:

- `counts_by_spectrum`, `spectrum_numbers`, `forward_counts`,
  `backward_counts`, `time_bins`, `good_frames`, `total_counts`;
- `events_millions`, `wall_seconds` and `time_bin_unit='ns'`.

Additional *synthetic* fields are `sample_fraction`, `rate_fraction`,
`expected_counts_per_frame`, `time_bin_edges`, `time_bin_width_ns`,
`expected_asymmetry`, `expected_polarisation`,
`expected_counts_by_spectrum`, `count_std_by_spectrum` and
`ideal_initial_asymmetry`. The scalar Bluesky Event includes expected
muons/frame and rate fraction; the large histograms stay in the in-memory
acquisition record. There is **no fabricated ISIS run number**.

Offline check (doesn't require EPICS/Bluesky or configured PVs):

```python
from functools import partial
from musrbeamsteer.synthetic import generate_synthetic_musr_data

simulate = partial(generate_synthetic_musr_data,
                   bounds=[(-1, 1), (-1, 1)],
                   center=[0, 0], curvature=[6, 6], delta_us_inv=0.75,
                   seed=1)
data = simulate([0, 0])
assert data['counts_by_spectrum'].shape == (64, 2000)
assert data['time_bin_edges'][-1] == 32000.0
```

## Hardware checkout

1. Confirm **actual** MUSR setpoint/readback PVs, correct signs, units,
   tolerances and safe simultaneous-current ranges.
2. Test just a few moves in narrow, approved bounds, with beam absent and
   `USE_SYNTHETIC=True`. Monitor actual readbacks and ensure no other
   scripting task moves the same magnets.
3. Remember that GP convergence warnings trigger random points anywhere
   inside the configured bounds; safety constraints must be enforced at the
   hardware layer as well.
4. Check interruptions and timeout handling. The optimiser attempts to
   restore starting currents on ordinary Python exceptions, but cannot
   guarantee restoration on external RunEngine halt or process termination.
5. Switch `USE_SYNTHETIC=False` only after actual MUSR DAE spectra mapping,
   t0, alpha, time units, detector corrections, EPICS lifecycle and saving
   behaviour have been checked on the instrument.

This project has local simulation tests, but is **not** hardware-qualified on
MUSR or EMU.


## Decimal precision and current bounds

The optimiser must never request a steering-current value outside
``SAFE_BOUNDS``. A fixed decimal precision can round a current outside
its original search range (for example, Python's ``round(-1.5, 0)``
is ``-2.0``). The Bluesky plan now clips rounded coordinates to the
configured bounds and independently validates them before any move.

``CURRENT_PRECISIONS = None`` now means **no rounding** (rather than
rounding to whole amps). For a device with a known setpoint increment,
configure an explicit precision, such as ``[3, 3]`` for milliamps
where setpoint units are amps. Clipping does not substitute for
validating physical hardware limits and readback tolerances.

If the two steering magnets are already outside the chosen search bounds
at the start of the plan, optimisation now refuses to move them. Check the
readbacks in IBEX and select safe, approved bounds that contain the
initial currents before starting the scan.

On Windows, a Channel Access message about a missing ``caRepeater.exe``
may appear if the EPICS Base executable directory is not on ``PATH``.
Check ``where caRepeater`` and ask the instrument controls team for the
correct EPICS tools location; do not confuse this diagnostic with
application-level bound errors.

## Live estimate of the minimum after each measurement

The MUSR steering example now defaults to:

```python
VERBOSE = 2                     # 0: quiet, 1: observed best, 2: + GP estimate
MINIMUM_ESTIMATE_POINTS = 1024  # applied only at VERBOSE=2
```

Once the initial design is complete (when PCA/global reductions and repeated
measurements can establish a noise estimate), every completed point prints:

```text
  8: x=[-0.006 -0.084], metric=..., noise=...
     best measured minimum: x=[... ...], metric=...
     GP estimated minimum: x=[... ...], predicted_metric=... (posterior mean, 1024 Sobol points + measurements)
```

The **best measured** coordinate is the point with the lowest current
observation metric. The **GP estimated minimum** is the lowest posterior-mean
prediction among Sobol candidate points *and* measured coordinates. It is an
approximate, possibly unmeasured location. Its predicted metric can be lower
than any measured metric; it is not a new acquired measurement. For PCA the
best measured metric is recomputed from the complete dataset at each step.

If a GP fit emits a warning such as `constant_value` or `length_scale` at a
bound, the display says `estimated minimum: best measured (GP fit warning;
model prediction not trusted)`. This does **not** disable the existing random
exploration recovery. The progress-only estimate does not change any device
position, proposed next coordinate, or the default `final_position="best_measured"`.

At `VERBOSE = 2`, calculating and displaying the GP estimate involves one
**additional GP fit per completed measurement** after the initial design;
reduce `MINIMUM_ESTIMATE_POINTS` to make the Sobol search faster, or choose
`VERBOSE = 1` to show the best measured point without those extra GP fits.
`VERBOSE = 0` disables routine optimiser progress output (safety/restoration
warnings remain visible). The initial-design points are printed as
`estimated minimum pending` at levels 1 and 2 until metrics can be calculated.

On a Windows installation where `caRepeater.exe` has been placed in the
current project directory, add that directory to the *current Command Prompt
session's* executable search path before running Python:

```cmd
cd /d "C:\Users\musr\Documents\Isaac\musr_beamsteer_bluesky"
set "PATH=%CD%;%PATH%"
where caRepeater.exe
```

The executable may require other EPICS Base DLLs; using the correct EPICS Base
installation's `bin` directory on `PATH` is preferable to copying the EXE on
its own. Missing `caRepeater.exe` is a separate CA diagnostic and should not
be confused with an optimisation error.
