import numpy as np
import pytest
from ..blueskysteercore.metrics import (IndependentMetric, GlobalMetric, count_rate_metric,
    initial_asymmetry_metric, asymmetry_squared_count_rate_metric)


def sample(uid, f, b, frames=10):
    return {'uid': uid, 'data': {'forward_counts': [f], 'backward_counts':[b],
            'total_counts': f+b, 'good_frames':frames}}


def test_metrics_and_cache():
    x = sample('x', 80, 20)
    y = sample('y', 60, 40)
    a = initial_asymmetry_metric()
    assert np.allclose(a([x,y]), [-0.6, -0.2])
    a([x,y])
    assert a.cache_info().hits == 2
    assert count_rate_metric()([x]) == [-10]
    assert np.isclose(asymmetry_squared_count_rate_metric()([x])[0], -3.6)


def test_new_measurement_only_computes_once():
    calls = []
    metric = IndependentMetric(lambda r: calls.append(r['uid']) or 1.0)
    x,y = sample('x',1,1),sample('y',2,2)
    metric([x]); metric([x,y]); metric([x,y])
    assert calls == ['x','y']


def test_global_recalculates():
    calls = []
    metric = GlobalMetric(lambda records, first_set: calls.append(len(records)) or [1]*len(records))
    metric([sample('a',1,1)]); metric([sample('a',1,1),sample('b',1,1)])
    assert calls == [1,2]


def test_zero_exposure_rejected():
    with pytest.raises(ValueError):
        count_rate_metric()([sample('x',1,1,0)])
