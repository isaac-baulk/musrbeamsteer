from ophyd_async.plan_stubs import ensure_connected
from bluesky.protocols import NamedMovable
import bluesky.plans as bp
import bluesky.plan_stubs as bps
from ibex_bluesky_core.callbacks import ISISCallbacks
from ibex_bluesky_core.fitting import Gaussian
from ibex_bluesky_core.devices.simpledae import monitor_normalising_dae
from ibex_bluesky_core.utils import centred_pixel
from ibex_bluesky_core.plan_stubs import with_num_periods


def detector_scan(block: NamedMovable, start, stop, num, frames=200):
    """
    Optimise the position of a block by fitting a Gaussian to the intensity on the detector,
    and moving the block to the optimum value.
    """
    dae = monitor_normalising_dae(
        det_pixels=centred_pixel(20, pixel_range=10),
        frames=frames,
        periods=True,
        save_run=False,
        monitor=2,
    )

    # Bluesky connects devices up-front for efficiency, and so that plans fail-fast if a
    # required PV is not available.
    yield from ensure_connected(block, dae)

    # When read, the DAE _may_ provide multiple readbacks per scan point.
    # Here, we tell bluesky to use the normalised intensity as the dependent variable,
    # and the corresponding standard deviations in plotting and fitting functions.
    icc = ISISCallbacks(
        x=block.name,
        y=dae.reducer.intensity.name,
        yerr=dae.reducer.intensity_stddev.name,
        fit=Gaussian().fit(),
    )

    # Apply the callbacks defined above to a "scan" command.
    @icc
    def _inner():
        # We delegate to bluesky's built-in absolute scan command.
        yield from bp.scan([dae], block, start, stop, num)

    # This ensures that the DAE is configured with sufficient DAE periods before the scan,
    # and then puts the number of DAE periods back to what it was before after the scan
    # (including if the scan failed with an exception).
    yield from with_num_periods(_inner(), dae=dae, number_of_periods=num)

    if icc.live_fit.result is not None:
        # Move to the optimum value
        yield from bps.mv(block, icc.live_fit.result.values["x0"])
    else:
        pass  # Handling for case where the fit failed