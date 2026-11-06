"""MUSR 2-D steering magnet test, with no muon beam / no DAE required.

Edit the CONFIGURATION section, especially the actual MUSR steering-magnet
EPICS setpoint/readback PVs, readback tolerances and approved current bounds.
The old ARGUS magnet PV definitions MUST NOT be reused here.

``USE_SYNTHETIC=True``: Bluesky drives the real TWO magnets, but the data are
simulated. The DAE is never connected, started or saved.
``USE_SYNTHETIC=False``: the SAME minimisation plan instead uses the live
MUSR DAE adapter. Every point runs BEGIN/WAIT/PAUSE/live-read/END+SAVE.

The synthetic signal has a position-dependent muon rate (10% at the
parameter-space boundary) and constant TRUE A(0). Both live-histogram
INTEGRATED ASYMMETRY and PCA metrics can locate the Kubo--Toyabe centre.
"""
from __future__ import annotations

from functools import partial
import ast
import re

import numpy as np

from bluesky import RunEngine

from musr_steering_main import make_plan


RE = RunEngine({})
RE(make_plan())