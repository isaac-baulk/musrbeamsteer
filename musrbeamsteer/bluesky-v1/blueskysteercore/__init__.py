"""Generic experimental optimisation with optional Bluesky / ISIS DAE support."""
__version__ = "0.4.0"
from .bluesky_integration import minimize_plan
from .metrics import (
    IndependentMetric, GlobalMetric, count_rate_metric,
    initial_asymmetry_metric, asymmetry_squared_count_rate_metric,
    legacy_metrics,
)
from .isis_dae import (
    AcquisitionDuration, DetectorGrouping, LiveSpectrumSnapshotter,
    NexusSpectrumSnapshotter,
    make_live_period_per_point_dae, make_live_run_per_point_dae,
    make_musr_live_dae, make_musr_live_dae_ppp,
    musr_longitudinal_grouping,
)

from .musr_metrics import integrated_asymmetry_metric, pca_asymmetry_metric
from .synthetic import (
    SyntheticMotor,
    generate_synthetic_musr_data,
    static_gaussian_kubo_toyabe,
    spatial_rate_fraction,
)
from .acquisition import make_synthetic_acquire, make_live_ibex_dae_acquire
from .positioners import make_setpoint_readback_axis
