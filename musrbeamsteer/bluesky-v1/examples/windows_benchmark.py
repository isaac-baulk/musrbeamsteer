"""
Windows benchmark for MUSR steering, comparing RunPerPoint and PeriodPerPoint

run with "python -m examples.windows_benchmark"

#1 run benchmark() This will peform 5 runs of each mode and
report the median and minimum times for each mode.

#2 change __main__ to prepare_cprofile() and run "python -m cProfile -o run_per_point.prof -m examples.windows_benchmark" to profile the run-per-point mode.
and "python -m cProfile -o period_per_point.prof -m examples.windows_benchmark" to profile the period-per-point mode.

then use "python -m pstats profile.prof" to view the profile and
"python -m pstats profile.prof -c" to view the profile sorted by cumulative time.

This will help identify time spent in optimisation, metric calculation, and Bluesky orchestration. 
It will not explain most time waiting on EPICS, DAE acquisition, or saving. For that,
enable benchmark_acq = True
"""



import time
import statistics
import musr_steering_main as steering
from bluesky import RunEngine

def run_once(period_per_point):
    steering.PERIOD_PER_POINT = period_per_point
    steering.benchmark_acq = True
    steering.PLOT_LIVE = False
    steering.VERBOSE = 0

    re = RunEngine({})
    plan = steering.make_plan()  # rebuild the generator each time

    start = time.perf_counter()
    result = re(plan)
    elapsed = time.perf_counter() - start
    steering.BenchmarkRecorder().log(
        "benchmark_run",
        test="run_once",
        period_per_point=period_per_point,
        elapsed_seconds=elapsed,
    )
    return elapsed, result

def benchmark_ppp_rpp():
    for mode in (False, True):
        samples = [run_once(mode)[0] for _ in range(5)]
        result = {
            "period_per_point": mode,
            "samples": samples,
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
        }
        print(result)
        steering.BenchmarkRecorder().log("benchmark_result", test="benchmark_ppp_rpp", **result)

def prepare_cprofile():
    mode = False  # False = run-per-point, True = period-per-point
    elapsed, _ = run_once(mode)
    result = {"period_per_point": mode, "elapsed_seconds": elapsed}
    print(result)
    steering.BenchmarkRecorder().log("benchmark_result", test="prepare_cprofile", **result)


if __name__ == "__main__":
    #benchmark_ppp_rpp() #uncomment to get a quick benchmark of both modes
    #prepare_cprofile()  # uncomment to run a single mode for profiling. will not measure time spent waiting on EPICS, DAE acquisition, or saving.
    pass
