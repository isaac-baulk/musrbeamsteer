"""Instrument example: N-D optimisation of a DAE count-rate metric.

Run *inside* a configured ISIS Bluesky environment. Adjust the prefix,
controller, devices, and event field mapping to your local IBEX deployment.
"""
from ophyd_async.plan_stubs import ensure_connected
from ibex_bluesky_core.devices.simpledae import (
    SimpleDae, RunPerPointController, GoodFramesWaiter, PeriodGoodFramesNormalizer,
)
from ibex_bluesky_core.utils import get_pv_prefix
from musrbeamsteer.bluesky_integration import minimize_plan
from musrbeamsteer.acquisition import make_ibex_dae_acquire
from musrbeamsteer.metrics import count_rate_metric


def make_plan(positioners, bounds, *, frames=500):
    prefix = get_pv_prefix()
    dae = SimpleDae(
        prefix=prefix,
        controller=RunPerPointController(save_run=True),
        waiter=GoodFramesWaiter(frames),
        reducer=PeriodGoodFramesNormalizer(prefix, detector_spectra=list(range(1,193))),
    )
    def plan():
        yield from ensure_connected(*positioners, dae, timeout=60)
        # Confirm these raw keys from the actual SimpleDae event documents.
        # The normaliser may provide intensity rather than raw total counts.
        def map_fields(data):
            return {'total_counts': data['YOUR_TOTAL_COUNTS_FIELD'],
                    'good_frames': data['YOUR_GOOD_FRAMES_FIELD']}
        acquire = make_ibex_dae_acquire(dae, transform=map_fields)
        return (yield from minimize_plan(
            motors=positioners, bounds=bounds, acquire=acquire,
            extract_metrics=count_rate_metric(),
            initial_repeats=2, max_measurements=30,
            final_position='best_measured',
        ))
    return plan()
