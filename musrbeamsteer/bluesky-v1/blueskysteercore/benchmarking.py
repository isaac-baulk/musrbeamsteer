"""Opt-in timing and configuration logging for MUSR benchmarks."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from time import time
from typing import Any


LOG_PATH = Path("benchmark_acq.log")


def duration_as_dict(duration: Any) -> dict[str, Any]:
    return {"unit": str(duration.unit), "value": duration.value}


class BenchmarkRecorder:
    def __init__(self, *, log_path: Path = LOG_PATH) -> None:
        self.log_path = log_path
        self.timings: dict[str, list[float]] = {}

    def log(self, event: str, **data: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            json.dump({"timestamp": time(), "event": event, **data}, stream, default=str)
            stream.write("\n")

    def record(self, name: str, seconds: float) -> None:
        self.timings.setdefault(name, []).append(seconds)
        self.log("timing", name=name, seconds=seconds)

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "count": len(values),
                "median_seconds": median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
            for name, values in self.timings.items()
            if values
        }

    def log_summary(self, *, print_run_per_point: bool = False) -> dict[str, dict[str, float | int]]:
        summary = self.summary()
        self.log("timing_summary", timings=summary)
        if print_run_per_point:
            names = (
                "dae_acquisition_wait",
                "dae_pause",
                "dae_snapshot",
                "dae_end_save",
            )
            print("Run-per-point median timings:")
            for name in names:
                if name in summary:
                    print(f"  {name}: {summary[name]['median_seconds']:.3f}s")
        return summary
