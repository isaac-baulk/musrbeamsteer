"""Physics, acquisition and end-to-end tests for the synthetic MUSR source."""
from __future__ import annotations

from functools import partial
import sys
import types

import numpy as np
import pytest

from ..blueskysteercore.isis_dae import AcquisitionDuration, musr_longitudinal_grouping
from ..blueskysteercore.synthetic import (
    generate_synthetic_musr_data, static_gaussian_kubo_toyabe,
    synthetic_good_frames_and_seconds, spatial_rate_fraction,
)
from ..blueskysteercore.metrics import count_rate_metric, initial_asymmetry_metric
from ..blueskysteercore.musr_metrics import integrated_asymmetry_metric, pca_asymmetry_metric
from ..blueskysteercore.acquisition import make_synthetic_acquire
from ..blueskysteercore.bluesky_integration import minimize_plan


BOUNDS = ((-1., 1.), (-1., 1.))
CENTRE = (0.2, -0.3)


def _simulator(rng=None, **params):
    return partial(
        generate_synthetic_musr_data,
        center=CENTRE, bounds=BOUNDS, curvature=[6.0, 5.0],
        delta_us_inv=0.75, initial_asymmetry=0.23,
        duration=AcquisitionDuration.good_frames(500),
        n_time_bins=256, max_time_us=10.0,
        rng=rng if rng is not None else np.random.default_rng(40),
        **params,
    )


def _consume(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def test_static_kt_exact_limits_and_dip():
    kt = static_gaussian_kubo_toyabe([0, 0.2, 2.3, 200], 0.75)
    assert kt[0] == pytest.approx(1)
    assert kt[1] < 1
    assert kt[2] < 1/3
    assert kt[-1] == pytest.approx(1/3)
    assert np.allclose(static_gaussian_kubo_toyabe([0, 1, 2], 0), 1)


def test_default_16_ns_binning_0_to_32_microseconds():
    snap = generate_synthetic_musr_data(
        [0.0], center=[0.0], bounds=[(-1., 1.)], curvature=[3.],
        delta_us_inv=0.75, seed=5,
    )
    assert snap['counts_by_spectrum'].shape == (64, 2000)
    assert snap['time_bin_width_ns'] == pytest.approx(16.)
    assert snap['time_bins'].shape == (2000,)
    assert snap['time_bins'][0] == pytest.approx(8.)
    assert snap['time_bins'][-1] == pytest.approx(31992.)
    assert snap['time_bin_edges'].shape == (2001,)
    assert snap['time_bin_edges'][0] == pytest.approx(0.)
    assert snap['time_bin_edges'][-1] == pytest.approx(32000.)
    assert np.allclose(np.diff(snap['time_bin_edges']), 16.)


def test_simulator_shapes_musr_groups_and_exponential():
    sim = _simulator()
    rec = sim(CENTRE)
    assert rec['counts_by_spectrum'].shape == (64, 256)
    assert list(rec['spectrum_numbers']) == list(range(1, 65))
    grouping = musr_longitudinal_grouping()
    rows = {spectrum: row for row, spectrum in enumerate(rec['spectrum_numbers'])}
    forward_rows = [rows[spectrum] for spectrum in grouping.forward]
    backward_rows = [rows[spectrum] for spectrum in grouping.backward]
    assert np.array_equal(rec['forward_counts'], rec['counts_by_spectrum'][forward_rows].sum(axis=0))
    assert np.array_equal(rec['backward_counts'], rec['counts_by_spectrum'][backward_rows].sum(axis=0))
    assert rec['total_counts'] == float(rec['counts_by_spectrum'].sum())
    assert rec['sample_fraction'] == pytest.approx(1)
    assert rec['rate_fraction'] == pytest.approx(1)
    assert rec['ideal_initial_asymmetry'] == pytest.approx(0.23)
    assert rec['time_bin_unit'] == 'ns'
    assert rec['good_frames'] == 500
    assert rec['synthetic']
    assert rec['expected_total_counts'] == pytest.approx(500_000)
    assert rec['expected_counts_by_spectrum'].sum() == pytest.approx(500_000)
    assert np.allclose(rec['count_std_by_spectrum']**2, rec['counts_by_spectrum'])
    assert rec['expected_counts_by_spectrum'][:, 0].sum() > 20 * rec['expected_counts_by_spectrum'][:, -1].sum()
    assert np.issubdtype(rec['counts_by_spectrum'].dtype, np.integer)


def test_rate_fraction_exact_on_each_box_edge_and_asymmetric_center():
    center = (0.2, -0.3)
    bounds = [(-1., 1.), (-2., 3.)]
    for edge in [(-1., -.3), (1., -.3), (.2, -2.), (.2, 3.),
                 (-1., 3.), (1., -2.), (-1., 0.1), (.5, -2.)]:
        assert spatial_rate_fraction(edge, center=center, bounds=bounds) == pytest.approx(0.1)
    assert spatial_rate_fraction(center, center=center, bounds=bounds) == pytest.approx(1)
    assert 0.1 < spatial_rate_fraction((.5, .0), center=center, bounds=bounds) < 1
    with pytest.raises(ValueError, match='strictly inside'):
        spatial_rate_fraction([1.], center=[1.], bounds=[(0., 1.)])
    with pytest.raises(ValueError, match='outside'):
        spatial_rate_fraction([2.], center=[0.], bounds=[(-1., 1.)])
    with pytest.raises(ValueError, match='edge_rate_fraction'):
        spatial_rate_fraction([0.], center=[0.], bounds=[(-1., 1.)], edge_rate_fraction=0)


def test_ideal_initial_asymmetry_constant_and_kt_centre_but_rate_falls():
    sim = _simulator()
    center = sim(CENTRE)
    edge = sim([1.0, -1.0])
    assert edge['sample_fraction'] < 0.01
    assert center['sample_fraction'] == 1
    assert center['ideal_initial_asymmetry'] == edge['ideal_initial_asymmetry'] == .23
    assert center['expected_asymmetry'][0] == pytest.approx(edge['expected_asymmetry'][0], abs=.0001)
    dip_ix = np.argmin(abs(center['time_bins'] - 2300))
    assert center['expected_asymmetry'][dip_ix] < .03
    assert edge['expected_asymmetry'][dip_ix] == pytest.approx(.23, abs=.002)
    assert edge['expected_counts_per_frame'] == pytest.approx(100.)
    assert edge['rate_fraction'] == pytest.approx(.1)
    assert edge['expected_total_counts'] == pytest.approx(center['expected_total_counts'] * .1)
    assert np.allclose(center['expected_counts_by_spectrum'].sum(axis=0) * .1,
                       edge['expected_counts_by_spectrum'].sum(axis=0))
    rate = count_rate_metric()
    measurements = [{'uid':'centre', 'data':center}, {'uid':'edge','data':edge}]
    assert rate(measurements)[0] < rate(measurements)[1]


def test_repeats_have_new_poisson_noise_and_seed_reproducible():
    a = _simulator(rng=np.random.default_rng(123))
    b = _simulator(rng=np.random.default_rng(123))
    first = a(CENTRE)['counts_by_spectrum']
    second = a(CENTRE)['counts_by_spectrum']
    assert not np.array_equal(first, second)
    assert np.array_equal(first, b(CENTRE)['counts_by_spectrum'])
    assert np.array_equal(second, b(CENTRE)['counts_by_spectrum'])


def test_poisson_std_matches_sqrt_expected():
    generator = np.random.default_rng(1)
    simulate = partial(
        generate_synthetic_musr_data,
        center=[0., 0.], bounds=BOUNDS, curvature=[1., 1.], delta_us_inv=.75,
        duration=AcquisitionDuration.good_frames(300), n_time_bins=32,
        max_time_us=6., rng=generator,
    )
    samples = np.array([simulate([0., 0.])['counts_by_spectrum'][0, 0] for _ in range(400)])
    expected = simulate([0., 0.])['expected_counts_by_spectrum'][0, 0]
    assert abs(samples.mean() - expected) / expected < .10
    assert np.sqrt(samples.var(ddof=1)) / np.sqrt(expected) == pytest.approx(1., abs=.13)


def test_duration_options_and_event_targets_use_local_rate():
    # Good-frame and wall-time limits keep exposures fixed while rates fall.
    for duration, expected_frames in [
        (AcquisitionDuration.good_frames(70), 70),
        (AcquisitionDuration.events(75_000), 75),
        (AcquisitionDuration.seconds(1.4), 70),
    ]:
        sim = partial(
            generate_synthetic_musr_data,
            center=[0.], bounds=[(-1., 1.)], curvature=[5.], delta_us_inv=.8,
            duration=duration, rng=np.random.default_rng(99),
            counts_per_frame=1000, good_frames_per_second=50,
        )
        snap = sim([0.])
        assert snap['good_frames'] == expected_frames
        assert snap['wall_seconds'] == pytest.approx(1.4 if expected_frames == 70 else 1.5)
        assert snap['events_millions'] == pytest.approx(snap['total_counts']/1e6)
        edge = sim([1.])
        assert edge['expected_counts_per_frame'] == pytest.approx(100.)
        if duration.unit.value == 'events':
            assert edge['good_frames'] == 750
            assert edge['expected_total_counts'] == pytest.approx(snap['expected_total_counts'])
        else:
            assert edge['good_frames'] == snap['good_frames']
            assert edge['expected_total_counts'] == pytest.approx(snap['expected_total_counts']*.1)


def test_integrated_asymmetry_and_pca_both_find_kt_centre():
    rng = np.random.default_rng(2)
    sim = partial(
        generate_synthetic_musr_data, center=[0., 0.], bounds=BOUNDS,
        curvature=[6., 6.], delta_us_inv=.75,
        duration=AcquisitionDuration.good_frames(5000), rng=rng,
    )
    records = [
        {'uid': 'centre', 'data': sim([0., 0.])},
        {'uid': 'middle', 'data': sim([.5, .5])},
        {'uid': 'edge', 'data': sim([1., 1.])},
    ]
    integrated = integrated_asymmetry_metric(time_window_ns=(0, 8000), rebin=8)
    values = integrated(records)
    assert values[0] < values[1] < values[2]
    assert integrated(records) == values
    assert integrated.cache_info().hits == 3
    pca = pca_asymmetry_metric(time_window_ns=(0, 8000), rebin=8)
    scores = pca(records)
    assert scores[0] < scores[1] < scores[2]
    assert pca.changes_with_dataset is True
    assert len(pca(records[:2])) == 2
    assert len(pca(records)) == 3
    # A(0) itself remains constant in expectation and cannot be a primary
    # objective here (finite-bin estimates of course have Poisson errors).
    initial = initial_asymmetry_metric(time_window=(0, 200))
    assert all(np.isfinite(initial(records)))


def test_invalid_parameters_fail_fast():
    sim = _simulator()
    with pytest.raises(ValueError, match='same nonzero'):
        sim([0.])
    with pytest.raises(ValueError, match='curvature'):
        generate_synthetic_musr_data([0.], center=[0.], bounds=[(-1., 1.)],
                                     curvature=[-2.], delta_us_inv=1.)
    with pytest.raises(ValueError, match='absolute'):
        generate_synthetic_musr_data([0.], center=[0.], bounds=[(-1., 1.)],
                                     curvature=[2.], delta_us_inv=1., initial_asymmetry=1.0)
    with pytest.raises(ValueError, match='rng OR seed'):
        generate_synthetic_musr_data([0.], center=[0.], bounds=[(-1., 1.)],
                                     curvature=[2.], delta_us_inv=1., rng=np.random.default_rng(), seed=1)
    with pytest.raises(ValueError, match='> 0'):
        synthetic_good_frames_and_seconds(AcquisitionDuration.seconds(1),
                                          counts_per_frame=-1, good_frames_per_second=50)


def fake_bluesky(monkeypatch):
    counter = {'runs': 0, 'events': [], 'close': [], 'sleep': []}

    def stub(result=None):
        if False:
            yield
        return result

    def open_run(md=None):
        counter['runs'] += 1
        counter.setdefault('metadata', []).append(md)
        return (yield from stub(f'synthetic-{counter["runs"]}'))

    def create(name='primary'):
        return (yield from stub(name))

    def read(dev):
        counter['events'].append((dev.describe(), dev.read()))
        return (yield from stub(dev.read()))

    def close_run(**kwargs):
        counter['close'].append(kwargs)
        return (yield from stub())

    def sleep(s):
        counter['sleep'].append(s)
        return (yield from stub())

    def mv(*args):
        for motor, value in zip(args[::2], args[1::2]):
            motor.value = float(value)
        return (yield from stub())

    def rd(motor):
        return (yield from stub(motor.value))

    stubs = types.SimpleNamespace(open_run=open_run, create=create, read=read,
                                  save=lambda: stub(), close_run=close_run,
                                  sleep=sleep, mv=mv, rd=rd)
    package = types.ModuleType('bluesky')
    package.__path__ = []
    package.plan_stubs = stubs
    monkeypatch.setitem(sys.modules, 'bluesky', package)
    monkeypatch.setitem(sys.modules, 'bluesky.plan_stubs', stubs)
    return counter


def test_synthetic_acquire_bluesky_documents_and_fresh_uid(monkeypatch):
    info = fake_bluesky(monkeypatch)
    simulator = _simulator(rng=np.random.default_rng(7))
    acquire = make_synthetic_acquire(simulator, delay_seconds=0.5)
    a = _consume(acquire(np.array([0.2, -0.3])))
    b = _consume(acquire(np.array([0.2, -0.3])))
    assert a['uid'] != b['uid']
    assert a['data']['synthetic'] is True
    assert 'run_number' not in a['data']  # no fake claim of saved ISIS run
    assert a['data']['counts_by_spectrum'].shape == (64, 256)
    assert not np.array_equal(a['data']['counts_by_spectrum'], b['data']['counts_by_spectrum'])
    assert len(info['events']) == 2
    assert info['metadata'][0]['synthetic'] is True
    assert info['metadata'][0]['optimisation_coordinate'] == [0.2, -0.3]
    assert info['sleep'] == [.5, .5]
    assert info['close'] == [{}, {}]
    d, r = info['events'][0]
    assert 'synthetic_dae_total_counts' in d
    assert r['synthetic_dae_good_frames']['value'] == 500
    assert r['synthetic_dae_rate_fraction']['value'] == pytest.approx(1.)


def test_synthetic_acquire_failure_closes_run_as_failed(monkeypatch):
    info = fake_bluesky(monkeypatch)
    def broken(coordinate):
        raise RuntimeError('simulator failure')
    acquire = make_synthetic_acquire(broken)
    with pytest.raises(RuntimeError, match='simulator failure'):
        _consume(acquire([0., 0.]))
    assert info['close'] == [{'exit_status': 'fail', 'reason': 'simulator failure'}]


@pytest.mark.parametrize("metric_factory", [integrated_asymmetry_metric, pca_asymmetry_metric])
def test_full_minimiser_with_fake_bluesky_and_synthetic_acquire(monkeypatch, metric_factory):
    info = fake_bluesky(monkeypatch)
    class Motor:
        def __init__(self, v): self.value = v
    motors = [Motor(.1), Motor(-.1)]
    simulator = partial(generate_synthetic_musr_data,
                        center=[0., 0.], bounds=BOUNDS, curvature=[8., 6.],
                        delta_us_inv=.75,
                        duration=AcquisitionDuration.good_frames(1000),
                        rng=np.random.default_rng(2026))
    acquire = make_synthetic_acquire(simulator)
    metric = metric_factory()
    result = _consume(minimize_plan(
        motors, [(-1., 1.), (-1., 1.)], acquire, metric,
        initial_coordinates=[(0., 0.), (-.5, .5)],
        initial_repeats=2, max_measurements=4,
        n_candidate_points=8, n_decision_points=16,
        precisions=[2, 2], verbose=False,
    ))
    assert result.n_measurements == 4
    assert len(info['events']) == 4
    assert all(o.measurement['synthetic'] for o in result.observations)
    assert all(np.isfinite(o.metric) for o in result.observations)
    best = min(result.observations, key=lambda obs: obs.metric)
    assert np.allclose([m.value for m in motors], best.coordinate)


def test_three_dimensional_centre_and_curvature():
    sim = partial(
        generate_synthetic_musr_data, center=[1.0, 2.0, -3.0],
        bounds=[(0., 3.), (0., 4.), (-5., -1.)],
        curvature=[1.0, 2.0, 0.5], delta_us_inv=.7,
        duration=AcquisitionDuration.good_frames(100),
        rng=np.random.default_rng(12),
    )
    centre = sim([1.0, 2.0, -3.0])
    displaced = sim([2.0, 3.0, -3.0])
    assert centre['sample_fraction'] == pytest.approx(1.)
    assert displaced['sample_fraction'] == pytest.approx(np.exp(-3.))
    assert centre['counts_by_spectrum'].shape == displaced['counts_by_spectrum'].shape


def test_example_requires_real_musr_pvs_and_safe_bounds(monkeypatch):
    from examples import musr_steering_main as example
    # The user-provided MUSR PVs/bounds are now populated. Check that the
    # original safeguards still reject unset or incomplete configurations.
    monkeypatch.setattr(example, 'STEER_1_SETPOINT_PV', 'REPLACE_WITH_MUSR_PV')
    with pytest.raises(ValueError, match='actual steering magnet'):
        example.make_plan()
    monkeypatch.setattr(example, 'STEER_1_SETPOINT_PV', 'IN:MUSR:KEPCO_04:CURRENT:SP')
    monkeypatch.setattr(example, 'SAFE_BOUNDS', ((None, None), (-1.5, 1.5)))
    with pytest.raises(ValueError, match='SAFE_BOUNDS'):
        example.make_plan()
