"""Simple EPICS positioners: write signed setpoint, wait for readback.

This module deliberately does NOT contain the ARGUS Danfysik polarity or
power sequencing: the two MUSR steering supplies described by the user need
only a current setpoint and a current readback. PV names are not guessed.
"""
from __future__ import annotations

import math
from typing import Literal


def _epics_pv(pv: str, default_transport: Literal["ca", "pva"]) -> str:
    """Select a deterministic EPICS transport unless PV explicitly specifies one."""
    if "://" in pv:
        scheme = pv.split("://", 1)[0]
        if scheme not in ("ca", "pva"):
            raise ValueError(f"Unsupported EPICS transport in PV {pv!r}")
        return pv
    return f"{default_transport}://{pv}"


def make_setpoint_readback_axis(
    *,
    name: str,
    setpoint_pv: str,
    readback_pv: str,
    tolerance: float,
    timeout: float = 30.0,
    limits: tuple[float, float] | None = None,
    protocol: Literal["ca", "pva"] = "ca",
):
    """Return an Ophyd Async Bluesky Settable/Readable positioner.

    Supply the exact EPICS PVs for the current MUSR configuration. No
    polarity/power commands are issued, and `bps.mv(axis, target)` waits
    until the measured current agrees to within `tolerance`.

    `protocol="ca"` is the default because MUSR instrument control PVs are
    normally accessed over EPICS Channel Access. `pva://` and `ca://` prefixes
    in supplied PVs take precedence. `ophyd-async[ca]` must be installed.

    `limits` independently bounds each command at the device level, even
    if the optimisation algorithm is misconfigured. Instrument-wide safety
    limits and interlocks must still be enforced in the hardware/IBEX layer.
    """
    if not name or not setpoint_pv or not readback_pv:
        raise ValueError("name, setpoint_pv and readback_pv are required")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and > 0")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and > 0")
    if limits is not None:
        lower, upper = map(float, limits)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("limits must be increasing finite bounds")
    else:
        lower, upper = float("-inf"), float("inf")
    if protocol not in ("ca", "pva"):
        raise ValueError("protocol must be 'ca' or 'pva'")
    resolved_setpoint_pv = _epics_pv(setpoint_pv, protocol)
    resolved_readback_pv = _epics_pv(readback_pv, protocol)

    from ophyd_async.core import AsyncStatus, HintedSignal, StandardReadable, wait_for_value
    from ophyd_async.epics.signal import epics_signal_r, epics_signal_w

    class SetpointReadbackAxis(StandardReadable):
        def __init__(self):
            with self.add_children_as_readables(HintedSignal):
                self.readback = epics_signal_r(float, resolved_readback_pv)
            self.setpoint = epics_signal_w(float, resolved_setpoint_pv)
            super().__init__(name=name)

        @AsyncStatus.wrap
        async def set(self, value: float) -> None:
            target = float(value)
            if not math.isfinite(target) or not lower <= target <= upper:
                raise ValueError(f"{name} target {target} outside configured limits {(lower, upper)}")
            await self.setpoint.set(target, timeout=timeout)
            await wait_for_value(
                self.readback,
                lambda actual: abs(float(actual) - target) <= tolerance,
                timeout=timeout,
            )

    return SetpointReadbackAxis()
