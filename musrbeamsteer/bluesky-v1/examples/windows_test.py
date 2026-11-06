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
    plan = steering.make_plan()
    from bluesky import RunEngine
    re = RunEngine({})
    return re(plan)

def plot_live_main():
    if steering.PLOT_LIVE:
        # Do not override a custom MPLBACKEND supplied by the operator.
        os.environ.setdefault("MPLBACKEND", "QtAgg")
    plan = steering.make_plan()  # prompts for guess and opens Qt figure on main thread
    from bluesky import RunEngine
    re = RunEngine({})
    result = re(plan)
    if steering.PLOT_LIVE:
        # Leave the final model/asymmetry visible for inspection after the
        # optimisation completes. Close the Qt window to exit the script.
        import matplotlib.pyplot as plt
        plt.show(block=True)
    return result


if __name__ == "__main__":
    main()