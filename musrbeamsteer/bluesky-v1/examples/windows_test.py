"""Windows-friendly MUSR steering entry point.

Run from the repository root::

    python -m examples.windows_test

This imports the package module correctly when launched with ``-m`` and then
runs the configured Bluesky plan.
"""
from __future__ import annotations

from . import musr_steering_main as steering
import os


def main():
    os.environ.setdefault("MPLBACKEND", "QtAgg")
    plan = steering.make_plan()
    from bluesky import RunEngine
    re = RunEngine({})
    return re(plan)


if __name__ == "__main__":
    main()