# MUSR steering magnets: EPICS Channel Access

The steering magnet positioner uses `ca://` explicitly by default. This avoids
accidentally selecting EPICS PVAccess when `p4p` is installed but Channel Access
(`aioca`) is absent. The previous PVA connection reported `Put permission denied
by gateway` for configured `IN:MUSR:KEPCO_04:CURRENT:SP` and
`IN:MUSR:KEPCO_05:CURRENT:SP`. Using CA is *not* a method of bypassing
access control. If the CA gateway also rejects writes, ask the instrument
controls team to check the authorised device and network access.

## Windows CMD (project already installed editable)

```
cd /d C:\Users\musr\Documents\Isaac\musr_beamsteer_bluesky
python -m pip install -e ".[bluesky]"
python -c "import aioca; print('CA client available')"
```

Check the intended magnet supplies and their actual values in IBEX **before**
allowing any motion. The PV names `KEPCO_04` / `KEPCO_05` should not be assumed
to be Danfysik supplies; confirm the labels and wiring.

Read-only test, if these are indeed the intended PVs:

```
python -c "import asyncio; from aioca import caget; print(asyncio.run(caget('IN:MUSR:KEPCO_04:CURRENT')))"
python -c "import asyncio; from aioca import caget; print(asyncio.run(caget('IN:MUSR:KEPCO_05:CURRENT')))"
```

On success, run the usual Bluesky plan only with verified safe ranges and
movement tolerances. **WARNING:** The synthetic acquisition produces fake
DAE data, but still commands real magnet movement.

```
python -c "from bluesky import RunEngine; from examples.musr_steering_synthetic import make_plan; RE = RunEngine({}); RE(make_plan())"
```

If the gateway still denies writes, do not attempt to work around those
permissions. Contact the instrument controls team. After any failed plan,
check both magnet readbacks in IBEX because one device may have moved even if
another failed. The plan's cleanup could not restore them in the original
error log.
