## New: ISIS live run-per-point acquisition

See [ISIS_LIVE_DAE.md](ISIS_LIVE_DAE.md) for the new DAE device that snapshots
live spectra before END/save (rather than post-END reducer processing),
supports `AcquisitionDuration.good_frames`, `.events` or `.seconds`, and
uses MUSR's manual detector groups as a configurable default.

# N-dimensional minimisation with Bluesky

```python
from musrbeamsteer.bluesky_integration import minimize_plan
from musrbeamsteer.metrics import count_rate_metric
result = RE(minimize_plan(
    motors=[axis_a, axis_b], bounds=[(-1,1), (10,20)],
    acquire=acquire_plan, extract_metrics=count_rate_metric(),
    initial_repeats=2, max_measurements=30))
```

`acquire_plan(x)` must be a **generator plan** returning a mapping with a
unique `uid` and a `data` mapping. `make_bluesky_acquire(detectors)` captures
Bluesky event documents inside the RunEngine, and
`make_ibex_dae_acquire(dae)` configures the same pattern for an IBEX SimpleDae.
Both require configured devices and an actual RunEngine.

## Synthetic motors

`SyntheticMotor` defines a named in-memory motor without connecting to EPICS.
The configured PV names are retained as metadata, while `safe_bounds` are
enforced before each move:

```python
from blueskysteercore.synthetic import SyntheticMotor

motor_config = SyntheticMotor(
    name="MUSR_STEER_1",
    setpoint_pv="IN:MUSR:KEPCO_04:CURRENT:SP",
    readback_pv="IN:MUSR:KEPCO_04:CURRENT",
    safe_bounds=(-0.5, 0.5),
    initial_value=0.0,
)
motor = motor_config.build()
```

Use the built devices in `minimize_plan`. The MUSR example uses these soft
motors when `USE_SYNTHETIC_MOTORS=True`; set it to `False` to use the
EPICS-backed positioners. Independently set `USE_SYNTHETIC_DATA` to choose
between simulated data and the live MUSR DAE.

## Gaussian initial designs (beam-off and generic experiments)

The generic `gaussian_initial_coordinates(bounds, initial_guess, ...)` helper
provides initial samples around an N-dimensional operator guess, without
modifying GP/KG or the ordinary uniform random warning recovery:

```python
from musrbeamsteer.gp_kg_minimizer import gaussian_initial_coordinates
initial = gaussian_initial_coordinates(
    bounds=[(-0.5, 0.5), (-1.5, 1.5)],
    initial_guess=[0.0, -0.2],
    precisions=[3, 3], edge_margin_fraction=0.02, random_state=20261023,
)
# Pass `initial_coordinates=initial` to minimize_plan() or search_minimum().
```

The Gaussian standard deviation on each axis is the **distance to its
nearest boundary divided by 3**. If the guess is very close to a boundary,
validation rejects it rather than creating a degenerate initial design.
Every point is drawn from the Gaussian, including the first; the optimiser
repeats that first sampled coordinate as required for noise estimation.
Out-of-range proposals are rejected (never clipped onto the boundary) and
samples are checked after optional precision rounding. The default number
is 1 + 2N unique points. These restrictions concern only the initial design;
subsequent GP/KG and warning-recovery proposals still use the full bounds.

The MUSR example prompts in Windows CMD for `INITIAL_GUESS` by default.
Set the constant to a numeric list to skip the prompt in scripts. See
[SYNTHETIC_MUSR.md](SYNTHETIC_MUSR.md).

## Metrics

All scores are **minimised**. By default the signal metrics negate their
figure of merit: `count_rate_metric`, `initial_asymmetry_metric`, and
`asymmetry_squared_count_rate_metric`. Pass `maximize=False` to minimise raw
values instead. Count rate is counts/good-frame by default. Initial asymmetry
is `(F-alpha B)/(F+alpha B)` over the first `bins` after time alignment;
A² count rate is `A**2*(F+B)/good_frames`. The last two need time-binned
forward and backward **counts**, not a scalar intensity. Apply detector
corrections, dead-time and t0 upstream. Map local DAE fields in `transform`.

`IndependentMetric` uses `functools.lru_cache` keyed by **measurement UID**:
old results do not change when new measurements arrive. A repeated measurement
at one coordinate gets a **new UID** and is evaluated independently.
`GlobalMetric` / `legacy_metrics` recomputes the whole history for PCA or
other metrics that depend on all acquired points.

## Operational notes

- The optimiser's GP/KG calculation is in `gp_kg_minimizer.py` and works for N
  coordinates; performance can become expensive in higher dimensions.
- The legacy NeXus/Mantid metric adapters still require an instrument-specific
  resolver mapping Bluesky UID to real ISIS run number; **UID != run number**.
- Check actual IBEX event field names. `examples/nd_minimisation.py` contains
  deliberately explicit `YOUR_...` placeholders rather than invented fields.
- On failure the plan attempts to restore starting motor positions; this cannot
  substitute for beamline interlocks, limits, or an emergency stop.
- Run `python -m pytest tests/test_metrics.py tests/test_bluesky_integration.py`.

## GP warning recovery: automatic random exploration

Both `minimize_plan` (Bluesky) and `search_minimum` (synchronous) automatically
change their next-point selection strategy if **any warning emitted by a
scikit-learn GaussianProcessRegressor fit** is detected. This includes
`ConvergenceWarning` messages mentioning `constant_value` or `length_scale`
reaching their bounds.

1. Fit the GP to all measurements and estimate the Knowledge Gradient (KG).
2. **If the fit warns**, discard that recommendation and sample a new random
   point uniformly from the N-dimensional bounds. The random candidate is
   rounded to `precisions`, is distinct from previously acquired coordinates,
   and obeys `minimum_step_sizes` between consecutive positions.
3. Acquire that point *through Bluesky* and compute its metric, updating the
   observation history (including recalculation of global/PCA metrics).
4. Refit the GP with the new observation. Continue choosing random points for
   each fit that warns. As soon as a fit produces no warnings, resume KG-based
   acquisition and the normal `accuracy` stopping criterion.

A random fallback deliberately **ignores `accuracy`**: the KG estimate is
unreliable while the GP warns. `max_measurements` remains a hard cap, so the
search cannot run indefinitely. If rounding and movement constraints leave no
unmeasured points, the search stops early.

A GP fit is also attempted at the end. If it still emits warnings, the result
uses the **best actually measured** coordinate and metric instead of relying on
an unreliable model prediction. This fallback also applies to
`final_position="model"` in the Bluesky plan.

The returned `SearchResult` includes:

- `random_explorations`: number of warning-triggered random acquisitions.
- `model_fit_ok`: false if the final model had warnings and the best-measured
  fallback was used.
- `gp_warnings`: the warning messages encountered during fitting.
- `converged`: false if the result ends with an unreliable GP model.
- `kg`: NaN if the latest next-point decision used random exploration, because
  no meaningful KG estimate is available for that decision.

**Operational caution:** Random points can be anywhere inside the entire
configured hyperrectangle. All supplied bounds must therefore be safe for the
hardware to visit, with device-level limits and interlocks independently
configured. Convergence warnings can also reflect a metric that is nearly
constant or uninformative; additional random measurements do not guarantee
that the warnings will disappear. `max_measurements` limits this behaviour.
The random sequence is reproducible for a fixed `random_state` and history.

For tests, run:

```bash
python -m pytest -q tests/test_metrics.py tests/test_bluesky_integration.py tests/test_gp_warning_recovery.py
```

Even if the KG fit is warning-free and suggests convergence, the final-model
fit is checked **before** stopping. If that final fit warns and there is room
in the measurement budget, another random acquisition is made and both fits
are retried with the new observation. The best-measured fallback is only used
when further sampling cannot continue (budget or search-space exhaustion).

## Per-measurement best-minimum diagnostics

Use the `verbose` setting with **both** `minimize_plan` and `search_minimum`:

| `verbose` | Console output | Extra GP fit for progress? |
|---|---|---|
| `0` | Quiet (safety/restoration warnings remain visible) | No |
| `1` | Measurement, best measured minimum, and fit/recovery warnings | No |
| `2` (default) | Level 1 + approximate GP minimum and next-proposal diagnostics | Yes |

For backwards compatibility, `verbose=False` means `0` and `verbose=True`
means `2`. The old `show_minimum_estimate` argument was removed: verbosity
now controls it. Initial acquisitions report an estimate pending until
all initial-design measurements have been collected. If the GP fit warns,
its predicted minimum is not trusted, and the best measured point is shown.
The random-exploration recovery strategy is unchanged.

`minimum_estimate_points=1024` adjusts the extra Sobol search at level 2.
Progress-only estimates **never** move the motors; the final Bluesky
destination still follows `final_position`.
