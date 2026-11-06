"""Windows-friendly MUSR steering entry point with Qt live monitoring.

Run from the repository root::

    python -m examples.run_musr_steering

This constructs the Matplotlib/Qt monitor on the main thread BEFORE creating
Bluesky's RunEngine. The RunEngine's default during_task handler can then keep
Qt responsive while the optimiser and EPICS/DAE plans execute elsewhere.
"""
from __future__ import annotations

import os

import musr_steering_main as steering


def main():
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