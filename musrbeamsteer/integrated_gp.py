"""Run a Gaussian-process beam-steering search using MuSR asymmetry data.

In normal operation, measurements are acquired through Genie.  When
``TEST_MODE`` is enabled, the acquisition callback selects the nearest
available dummy Nexus run and copies it into the normal input directory so
the rest of the reduction and optimisation pipeline remains unchanged.
"""

import os
import time
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

from mantid.simpleapi import *
from mantid.api import AlgorithmManager
from mantid.kernel import ConfigService
from genie_python import genie as g

from gp_kg_minimizer import Observation, search_minimum
from TT_PCA.data_handler import DataHandler
from TT_PCA.pca_machine import PCAMachine

# ==========================================================================
# USER CONFIGURATION
# ==========================================================================

TEST_MODE = False
PROCESSED_RUNS_LOG = "../data/processed/processed_runs.log"
CALCULATE_ALPHA = False
SUPPRESS_MANTID_STATUS_LOGS = True
LONGITUDINAL_MODE = True  # Set to False for transverse mode

SCAN_PVS = [
    {"name": "hsm_steering", "pv": "IN:MUSR:KEPCO_04:CURRENT:SP", "bounds": (-0.5, 0.5), "precision": 2, "minimum_step_size": 0.01},
    {"name": "vsm_steering", "pv": "IN:MUSR:KEPCO_05:CURRENT:SP", "bounds": (-1.5, 1.5), "precision": 2, "minimum_step_size": 0.01},
    # Add other PVs here
]

INITIAL_COORDINATES = [(-0.20,  0.20), (-0.20, -0.20), (0.20, 0.20), (0.20, -0.20),(0.00,0.00),(0.10,-0.60)]
MEVENTS_PER_MEASUREMENT = 1
SETTLE_SECONDS = 2.0
ACCURACY = 0.00001
INITIAL_REPEATS = 2
MAX_MEASUREMENTS = 40
LAST_GOOD_VALUE = 14

# ==========================================================================
# SETUP
# ==========================================================================

if TEST_MODE:
    INPUT_DIR = "../data/incoming_nexus"
    OUTPUT_DIR = "../data/processed_nexus"
    ALPHA_FILE = None
    DUMMY_NEXUS_DIR = "../data/dummy_nexus"

    dummy_run_files = sorted(Path(DUMMY_NEXUS_DIR).glob("musr*.nxs"))
    if not dummy_run_files:
        raise FileNotFoundError(
            f"No dummy Nexus files found in {DUMMY_NEXUS_DIR!r}."
        )
    first_dummy_run = re.fullmatch(
        r"musr(\d+)\.nxs",
        dummy_run_files[0].name,
        re.IGNORECASE,
    )
    if first_dummy_run is None:
        raise ValueError(
            f"First dummy Nexus file has an invalid run name: "
            f"{dummy_run_files[0].name!r}."
        )
    START_RUN_NUMBER = int(first_dummy_run.group(1).lstrip("0") or "0")

else:
    INPUT_DIR = "//musr/Data"
    OUTPUT_DIR = "../data/processed_nexus"
    ALPHA_FILE = None
    DUMMY_NEXUS_DIR = None
    g.set_instrument("NDXMUSR")
    START_RUN_NUMBER = int(g.get_runnumber())
    print('Starting from run number:', START_RUN_NUMBER)

if SUPPRESS_MANTID_STATUS_LOGS:
    ConfigService["logging.loggers.root.level"] = "error" 

BOUNDS = [item["bounds"] for item in SCAN_PVS]
MINIMUM_STEP_SIZES = [item["minimum_step_size"] for item in SCAN_PVS]
PRECISIONS = [item["precision"] for item in SCAN_PVS]

if LONGITUDINAL_MODE:
    FORWARD_SPECTRA = list(range(33,65))  
    BACKWARD_SPECTRA = list(range(1,33))
    FIRST_INDEX = 0
    SECOND_INDEX = 1
else:
    FORWARD_SPECTRA = list(range(9,17)) + list(range(57, 65))
    BACKWARD_SPECTRA = list(range(25, 33)) + list(range(41,49))
    FIRST_INDEX = 3
    SECOND_INDEX = 1

# cache ===================================================
_bin_params_cache = None
_dummy_measurement_files = None
_dummy_measurement_paths = set()
_next_test_run_number = 900000

# ==========================================================================
# MANTID & ASYMMETRY FUNCTIONS
# ==========================================================================
def get_parameters(ws: Any, round_flag: bool = False) -> dict[str, float]:
    """Return the configured steering PV averages recorded in a workspace.

    Parameters
    ----------
    ws : Any
        Mantid workspace containing the steering process-variable logs.
    round_flag : bool, default=False
        Whether values should be rounded to two decimal places.
    Returns
    -------
    dict[str, float]
        A mapping from configured PV names to available mean values.
    """

    run = ws.getRun()

    def get_time_avg(log_name: str) -> float | None:
        """Return the time mean for one workspace log, if available.

        Parameters
        ----------
        log_name : str
            Name of the workspace log property.
        Returns
        -------
        float or None
            The log's time mean, or ``None`` when Mantid cannot provide it.
        """
        try:
            return run.getProperty(log_name).getStatistics().time_mean
        except RuntimeError:
            return None
    raw_parameters = {pv["name"]: get_time_avg(pv["name"]) for pv in SCAN_PVS}
    if round_flag:
        raw_parameters = {k: round(v, 2) for k, v in raw_parameters.items()}
    return {k: v for k, v in raw_parameters.items() if v is not None}

def get_alpha(
    calibration_file: str | os.PathLike[str],
    FORWARD_SPECTRA: Sequence[int],
    BACKWARD_SPECTRA: Sequence[int],
    last_good_value: float,
) -> Any:
    """Calculate the detector-pair asymmetry correction from a calibration run.

    Parameters
    ----------
    calibration_file : str or os.PathLike[str]
        Calibration Nexus file path.
    FORWARD_SPECTRA : sequence of int
        Forward detector spectrum numbers.
    BACKWARD_SPECTRA : sequence of int
        Backward detector spectrum numbers.
    last_good_value : float
        End of the valid calibration time range.
    Returns
    -------
    Any
        The Mantid detector-pair asymmetry correction.
    """
    c_ws = LoadMuonNexus(Filename=calibration_file, OutputWorkspace="a_ws", DetectorGroupingTable="grouping", DeadTimeTable='deadtimes')
    return AlphaCalc(InputWorkspace='a_ws', ForwardSpectra=FORWARD_SPECTRA, BackwardSpectra=BACKWARD_SPECTRA, FirstGoodValue=c_ws[3], LastGoodValue=last_good_value)

def calculate_constant_error_bins(
    workspace: Any,
    start_time: float,
    end_time: float,
    target_counts: float,
    FORWARD_SPECTRA: Sequence[int],
    BACKWARD_SPECTRA: Sequence[int],
    time_zero: float,
) -> list[float]:
    """Build Mantid rebin parameters with approximately constant count errors.

    Parameters
    ----------
    workspace : Any
        Mantid workspace containing detector histograms.
    start_time : float
        First time included in the rebinning range.
    end_time : float
        Last time included in the rebinning range.
    target_counts : float
        Minimum accumulated counts per generated bin.
    FORWARD_SPECTRA : sequence of int
        Forward detector spectrum numbers.
    BACKWARD_SPECTRA : sequence of int
        Backward detector spectrum numbers.
    time_zero : float
        Time-zero offset to subtract from the X data.
    Returns
    -------
    list of float
        Mantid-compatible ``[start, step, end, ...]`` rebin parameters.
    """

    x_data = workspace.readX(0) - time_zero
    f_idx = [workspace.getIndexFromSpectrumNumber(s) for s in FORWARD_SPECTRA]
    b_idx = [workspace.getIndexFromSpectrumNumber(s) for s in BACKWARD_SPECTRA]
    y_total = sum(workspace.readY(i) for i in f_idx) + sum(workspace.readY(i) for i in b_idx)
    
    new_bins = [start_time]
    current_counts = 0
    for i in range(len(y_total)):
        if start_time <= x_data[i] and x_data[i+1] <= end_time:
            current_counts += y_total[i]
            if current_counts >= target_counts:
                new_bins.append(x_data[i+1])
                current_counts = 0
    if new_bins[-1] != end_time:
        new_bins.append(end_time)
        
    mantid_params = []
    for i in range(len(new_bins) - 1):
        start, end = new_bins[i], new_bins[i+1]
        step = end - start
        if i == 0:
            mantid_params.extend([start, step, end])
        else:
            mantid_params.extend([step, end])
    return mantid_params

def get_asymmetry(
    nexus_file_path: str | os.PathLike[str],
    alpha: Any,
) -> tuple[Any, dict[str, float]]:
    """Load one Nexus run and convert it into a paired asymmetry workspace.

    Parameters
    ----------
    nexus_file_path : str or os.PathLike[str]
        Nexus run to load.
    alpha : Any
        Detector-pair asymmetry correction.
    Returns
    -------
    tuple[Any, dict[str, float]]
        A Mantid asymmetry workspace and the run's steering parameters.
    """

    global _bin_params_cache
    load_result = LoadMuonNexus(Filename=nexus_file_path, OutputWorkspace="test_result", DetectorGroupingTable="grouping", DeadTimeTable='deadtimes')
    loaded_time_zero, loaded_time_good = load_result[2], load_result[3]
    shifted_time_good = loaded_time_good - loaded_time_zero
    shifted_last_good = LAST_GOOD_VALUE - loaded_time_zero
    params = get_parameters(load_result[0])
        
    if _bin_params_cache is None:
        _bin_params_cache = calculate_constant_error_bins(load_result[0], shifted_time_good, shifted_last_good,
                                                          5000, FORWARD_SPECTRA, BACKWARD_SPECTRA, loaded_time_zero)

    f_b_counts = MuonProcess(
        InputWorkspace=load_result[0], OutputWorkspace="ws_asym", Mode="Combined",
        DetectorGroupingTable='grouping', SummedPeriodSet="1", LoadedTimeZero=loaded_time_zero,
        OutputType="PairAsymmetry", DeadTimeTable='deadtimes', Xmin=shifted_time_good,
        Xmax=shifted_last_good, Alpha=alpha, ApplyDeadTimeCorrection=True,
        PairFirstIndex=FIRST_INDEX, PairSecondIndex=SECOND_INDEX, RebinParams=_bin_params_cache
    )
    return f_b_counts, params

def _add_dummy_asymmetry_noise(
    asym_data: Sequence[float],
    error_data: Sequence[float],
) -> np.ndarray:
    """Add one Gaussian measurement-noise realization to asymmetry data.

    Parameters
    ----------
    asym_data : sequence of float
        Asymmetry values to perturb.
    error_data : sequence of float
        Standard deviation for each value.
    Returns
    -------
    numpy.ndarray
        A NumPy array containing the perturbed asymmetry values.
    """

    asym_data = np.asarray(asym_data, dtype=float)
    error_data = np.asarray(error_data, dtype=float)
    return asym_data + np.random.normal(0.0, error_data, size=asym_data.shape)

def process_batch(alpha: Any) -> None:
    """Reduce new Nexus files after ``START_RUN_NUMBER`` without modifying inputs.

    Parameters
    ----------
    alpha : Any
        Detector-pair asymmetry correction passed to Mantid.
    Raises
    ------
    OSError
        If an input or output file cannot be read or written.
    """
    
    processed_log_path = Path(OUTPUT_DIR) / PROCESSED_RUNS_LOG
    processed_runs = set()
    if processed_log_path.exists():
        with processed_log_path.open("r", encoding="utf-8") as log_file:
            for line in log_file:
                line = line.strip()
                if line:
                    processed_runs.add(int(line))

    run_pattern = re.compile(r"^musr(\d+)\.nxs$")
    entries = []
    with os.scandir(INPUT_DIR) as directory_entries:
        for entry in directory_entries:
            match = run_pattern.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                continue
            run_number = int(match.group(1))
            if run_number < START_RUN_NUMBER or run_number in processed_runs:
                continue
            entries.append((run_number, entry))

    for run_number, entry in sorted(entries, key=lambda item: item[0]):
        nxs_path = entry.path
        out_path = os.path.join(OUTPUT_DIR, f"{entry.name[:-4]}.txt")
        ws, params = get_asymmetry(nxs_path, alpha)

        time_data, asym_data, error_data = ws.extractX()[0], ws.extractY()[0], ws.extractE()[0]
        if Path(nxs_path) in _dummy_measurement_paths:
            asym_data = _add_dummy_asymmetry_noise(asym_data, error_data)
        with open(out_path, 'w') as output_file:
            valid_params = {k: v for k, v in params.items() if v is not None}
            if valid_params:
                output_file.write(f"# {'_'.join([f'{k}={v:g}' if isinstance(v, (int, float)) else f'{k}={v}' for k, v in valid_params.items()])}\n")
            for t, a, e in zip(time_data, asym_data, error_data):
                output_file.write(f"{t}\t{a}\t{e}\n")

        with processed_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{run_number}\n")
        processed_runs.add(run_number)


def _wait_for_measurement_run(measurements: Sequence[Mapping[str, Any]]) -> None:
    """Wait until the most recently requested Nexus run is available.

    Parameters
    ----------
    measurements : sequence of mapping
        Measurement records whose final ``run_number`` identifies the newest
        expected acquisition.
    Raises
    ------
    ValueError
        If the final measurement does not contain a valid run number.
    """
    run_text = str(measurements[-1]["run_number"])
    try:
        expected_run = int(run_text.lstrip("0") or "0")
    except ValueError as error:
        raise ValueError(
            f"Invalid run number in final measurement: {run_text!r}."
        ) from error

    run_pattern = re.compile(r"^musr(\d+)\.nxs$", re.IGNORECASE)
    input_path = Path(INPUT_DIR)
    while True:
        for file_path in input_path.glob("musr*.nxs"):
            match = run_pattern.fullmatch(file_path.name)
            if match is not None and int(match.group(1)) == expected_run:
                return
        time.sleep(1)

# ==========================================================================
# PCA & METRIC EXTRACTION
# ==========================================================================

def _sync_and_process_batch(measurements: Sequence[Mapping[str, Any]], alpha: Any) -> None:
    """Synchronize the input directory and process any new Nexus files.

    Parameters
    ----------
    measurements : sequence of mapping
        Measurement records whose final ``run_number`` identifies the newest
        expected acquisition.
    alpha : Any
        Detector-pair asymmetry correction passed to Mantid.
    """
    _wait_for_measurement_run(measurements)
    process_batch(alpha)

def get_parameters_from_files(
    stsp: tuple[int, int],
    prenum: str,
    ext: str,
) -> list[dict[str, float | int]]:
    """Read steering metadata written in processed text-file headers.

    Parameters
    ----------
    stsp : tuple[int, int]
        Inclusive first and last run numbers to read.
    prenum : str
        Text prefix before each run number.
    ext : str
        File extension, including its leading dot.
    Returns
    -------
    list[dict[str, float | int]]
        One metadata mapping per run, including its integer run number.
    """

    metadata = []
    pattern = re.compile(r"([a-zA-Z][a-zA-Z_]*)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    for run_number in range(stsp[0], stsp[1] + 1):
        file_path = Path(OUTPUT_DIR) / f"{prenum}{run_number}{ext}"
        with file_path.open("r", encoding="utf-8") as file:
            title = file.readline().strip()
        matches = pattern.findall(title)
        file_metadata = {"run": run_number}
        for key, val in matches:
            file_metadata[key] = float(val)
        metadata.append(file_metadata)
    return metadata

def integrated_asymmetry(
    measurements: Sequence[Mapping[str, Any]],
) -> list[float]:
    """Return one inverse-variance-weighted asymmetry value per measurement.

    Each processed text file contains one row per time bin with columns
    ``time, asymmetry, error``.  A time bin with uncertainty ``sigma_i`` is
    weighted by ``1 / sigma_i**2``::

        A_weighted = sum(A_i / sigma_i**2) / sum(1 / sigma_i**2)

    The returned list follows the exact order of ``measurements``, as required
    by ``search_minimum(..., extract_metrics=...)``.

    Parameters
    ----------
    measurements : sequence of mapping
        Measurement records containing a ``run_number`` field.
    Returns
    -------
    list of float
        One inverse-variance-weighted asymmetry per measurement.
    Raises
    ------
    ValueError
        If a requested run or usable data column is missing.
    """
    if not measurements:
        return []

    # Process any newly acquired Nexus files into the text files containing
    # time, asymmetry, and error columns.
    if CALCULATE_ALPHA:
        alpha = get_alpha(
            ALPHA_FILE,
            FORWARD_SPECTRA,
            BACKWARD_SPECTRA,
            LAST_GOOD_VALUE,
        )
    else:
        alpha = 1

    _wait_for_measurement_run(measurements)
    process_batch(alpha)

    # Map run number -> processed text file.  Converting the run number to int
    # makes this insensitive to leading zeros in either source.
    run_files = {}
    for file_path in Path(OUTPUT_DIR).glob("*.txt"):
        match = re.fullmatch(r"musr(\d+)\.txt", file_path.name, re.IGNORECASE)
        if match is not None:
            run_files[int(match.group(1))] = file_path

    metrics = []

    for measurement in measurements:
        run_text = str(measurement["run_number"])
        run_number = int(run_text.lstrip("0") or "0")

        file_path = run_files.get(run_number)
        if file_path is None:
            raise ValueError(
                f"Run {run_text} requested by optimizer but no processed "
                f"asymmetry file was found in {OUTPUT_DIR!r}."
            )

        data = np.loadtxt(file_path, comments="#")
        data = np.atleast_2d(data)
        if data.shape[1] < 3:
            raise ValueError(
                f"Expected time, asymmetry, and error columns in {file_path}."
            )

        asymmetry = data[:, 1]
        errors = data[:, 2]

        # Ignore unusable bins rather than allowing NaN/Inf or a zero error bar
        # to dominate the metric.
        valid = (
            np.isfinite(asymmetry)
            & np.isfinite(errors)
            & (errors > 0)
        )
        if not np.any(valid):
            raise ValueError(
                f"Run {run_text} contains no finite bins with positive errors."
            )

        weights = 1.0 / np.square(errors[valid])
        weighted_asymmetry = np.sum(asymmetry[valid] * weights) / np.sum(weights)
        metrics.append(float(weighted_asymmetry))

    return metrics
    
def metric_function(
    measurements: Sequence[Mapping[str, Any]],
) -> list[float]:
    """Process all measurements and return one PCA metric per optimizer run.

    Parameters
    ----------
    measurements : sequence of mapping
        Measurement records containing a ``run_number`` field.
    Returns
    -------
    list of float
        One PCA score per measurement, in input order.
    Raises
    ------
    ValueError
        If output data are unavailable or a run is not in PCA output.
    """
    
    # 1. Process any newly acquired Nexus files into text files
    if CALCULATE_ALPHA == True:
        alpha = get_alpha(ALPHA_FILE, FORWARD_SPECTRA, BACKWARD_SPECTRA, LAST_GOOD_VALUE)
    else:
        alpha = 1
     
    _wait_for_measurement_run(measurements)
    process_batch(alpha)

    # 2. Extract bounds for batch loading
    files = sorted(
        f for f in Path(OUTPUT_DIR).iterdir()
        if f.is_file() and f.suffix == ".txt"
    )
    if not files:
        raise ValueError("No text files found in output directory.")
    
    match_first = re.match(r"^([a-zA-Z]+)(\d+)\.(.+)$", files[0].name)
    match_last = re.match(r"^([a-zA-Z]+)(\d+)\.(.+)$", files[-1].name)
    letters, first_numbers, extension = match_first.groups()
    _, last_numbers, _ = match_last.groups()
    prenum = letters + "0" * (len(first_numbers) - len(first_numbers.lstrip("0")))
    print(first_numbers)
    print(last_numbers)
    print(prenum)
    stsp = (int(first_numbers), int(last_numbers))
    ext = '.' + extension

    # 3. Load data and run PCA
    dh = DataHandler()
    dh.load_batch(loc=OUTPUT_DIR + '/', stsp=stsp, prenum=prenum, ext=ext, skiprows=1)
    pvs = get_parameters_from_files(stsp, prenum, ext)
    
    dh.prepare_XYE_PCA()
    pca_machine = PCAMachine(dh)
    pca_machine.perform_pca()
    
    # PCA/SVD eigenvectors have an arbitrary sign: the same PC can be returned
    # as either +v or -v on different recomputations.  Fix a deterministic
    # orientation using the whole PC1 curve, weighted by the statistical error
    # in each prepared time bin.  Inverse-variance weighting prevents a
    # low-statistics/high-error bin from determining the sign.
    pc1_curve = pca_machine.pc_curves[0][:, 0]
    pc1_errors = np.asarray(dh.prepared_data[0][2], dtype=float)

    valid = (
        np.isfinite(pc1_curve)
        & np.isfinite(pc1_errors)
        & (pc1_errors > 0)
    )
    if not np.any(valid):
        raise ValueError(
            "Cannot determine the PC1 sign: no finite prepared bins with "
            "positive error bars."
        )

    weights = 1.0 / np.square(pc1_errors[valid])
    weighted_pc1_sum = np.sum(pc1_curve[valid] * weights)

    if weighted_pc1_sum < 0:
        pca_machine.pc_curves[0][:, 0] *= -1
        pca_machine.pc_scores[0][0, :] *= -1

    pc1_scores = pca_machine.pc_scores[0][0, :].copy()

    # 4. Map the PCA scores back to the exact order the optimizer requested
    score_map = {str(pv_data["run"]): score for pv_data, score in zip(pvs, pc1_scores)}
    
    metrics = []
    for m in measurements:
        run_str = str(m["run_number"]).lstrip("0")
        if run_str in score_map:
            metrics.append(score_map[run_str])
        else:
            raise ValueError(f"Run {run_str} requested by optimizer but not found in PCA output.")
            
    return metrics

# ==========================================================================
# OPTIMIZER CONTROL
# ==========================================================================
def _get_dummy_measurement_file(
    coordinate: np.ndarray,
) -> Path:
    """Select and apply the nearest available dummy-run coordinate.

    Dummy run logs are rounded to the precision represented by the test data.
    The coordinate is updated in place because the minimizer retains the same
    NumPy array as the observation coordinate after this callback returns.

    Parameters
    ----------
    coordinate : numpy.ndarray
        Requested coordinate, updated in place to the selected run.
    Returns
    -------
    pathlib.Path
        Path to the nearest matching dummy Nexus file.
    Raises
    ------
    FileNotFoundError
        If no dummy run has all required parameters.
    """

    global _dummy_measurement_files

    if _dummy_measurement_files is None:
        _dummy_measurement_files = []
        for source_path in sorted(Path(DUMMY_NEXUS_DIR).glob("*.nxs")):
            if ALPHA_FILE is not None:
                if source_path == Path(ALPHA_FILE):
                    continue
            workspace_name = f"dummy_{source_path.stem}"
            workspace = LoadMuonNexus(
                Filename=str(source_path),
                OutputWorkspace=workspace_name,
                DetectorGroupingTable="grouping",
                DeadTimeTable="deadtimes",
            )[0]
            parameters = get_parameters(workspace, round_flag=True)
            print(f"Loaded {source_path.name} with parameters: {parameters}")
            _dummy_measurement_files.append((source_path, parameters))

    parameter_names = [item["name"] for item in SCAN_PVS]
    requested_coordinate = np.asarray(coordinate, dtype=float)
    available_files = [
        (source_path, parameters)
        for source_path, parameters in _dummy_measurement_files
        if all(name in parameters for name in parameter_names)
    ]
    if not available_files:
        raise FileNotFoundError(
            "No dummy Nexus files contain all required parameters: "
            + ", ".join(parameter_names)
        )

    source_path, parameters = min(
        available_files,
        key=lambda item: np.sum(
            (
                requested_coordinate
                - np.asarray([item[1][name] for name in parameter_names])
            ) ** 2
        ),
    )
    coordinate[:] = [
        parameters[name]
        for name in parameter_names
    ]
    return source_path


def _get_test_destination() -> tuple[Path, str]:
    """Return a unique input path and run number for test-mode acquisition.

    Returns
    -------
    tuple[pathlib.Path, str]
        A destination path and its run number as a string.
    """

    global _next_test_run_number

    while True:
        run_number = _next_test_run_number
        _next_test_run_number += 1
        destination_path = Path(INPUT_DIR) / f"musr{run_number:08d}.nxs"
        archived_path = destination_path.with_name(f"processed_{destination_path.name}")
        if not destination_path.exists() and not archived_path.exists():
            return destination_path, str(run_number)

def get_measurement(coordinate: Sequence[float]) -> dict[str, object]:
    """Acquire one measurement through Genie or a matching dummy Nexus run.

    Parameters
    ----------
    coordinate : sequence of float
        Steering coordinate in the configured PV order.
    Returns
    -------
    dict[str, object]
        A measurement record containing run number, coordinate, and event count.
    Raises
    ------
    ValueError
        If the coordinate dimension is invalid.
    """

    if len(coordinate) != len(SCAN_PVS):
        raise ValueError(
            "Coordinate dimension does not match the number of SCAN_PVS."
        )

    if TEST_MODE:
        # Reusing the source is safe because every copy gets a unique run name.
        source_path = _get_dummy_measurement_file(coordinate)
        destination_path, run_number = _get_test_destination()
        shutil.copy2(source_path, destination_path)
        _dummy_measurement_paths.add(destination_path)
        return {
            "run_number": run_number,
            "coordinate": tuple(float(v) for v in coordinate),
            "mevents": MEVENTS_PER_MEASUREMENT,
        }

    for item, value in zip(SCAN_PVS, coordinate):
        precision = item.get("precision", 2)
        value = round(value, precision)
        g.set_pv(item["pv"], float(value), wait=True, is_local=False)
    
    if SETTLE_SECONDS > 0:
        g.waitfor_time(seconds=SETTLE_SECONDS, quiet=True)

    title_parts = []
    for item, value in zip(SCAN_PVS, coordinate):
        title_parts.append(f"{item['name']}={float(value):+.6g}")
    title = ", ".join(title_parts)
    g.change_title(f"GP/KG scan: {title}")
    
    g.begin(quiet=True)
    run_number = str(g.get_runnumber())
    g.waitfor(mevents=MEVENTS_PER_MEASUREMENT, quiet=False)
    actual_mevents = float(g.get_mevents())
    g.end(quiet=True)

    return {"run_number": run_number, "coordinate": tuple(float(v) for v in coordinate), "mevents": actual_mevents}

def show_progress(
    observations: Sequence[Observation],
    estimated_noise_std: float | None,
) -> None:
    """Print the current metric and coordinate for every observation.

    Parameters
    ----------
    observations : sequence of Observation
        Optimizer observations to display.
    estimated_noise_std : float or None
        Current estimated measurement noise, if available.
    """

    print("\nCurrent global metrics:")
    for index, obs in enumerate(observations):
        vals = ", ".join(f"{item['name']}={value:+.6g}" for item, value in zip(SCAN_PVS, obs.coordinate))
        print(f"  {index:2d}  run={obs.measurement['run_number']}  {vals}  metric={obs.metric:.8g}")

if __name__ == "__main__":
    # The reduction callback expects both directories to exist before the
    # optimizer begins acquiring its initial design.
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start every run with an empty output directory and processed-runs log.
    output_path = Path(OUTPUT_DIR)
    for entry in output_path.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    result = search_minimum(
        accuracy=ACCURACY,
        bounds=BOUNDS,
        minimum_step_sizes=MINIMUM_STEP_SIZES,
        precisions=PRECISIONS,
        get_measurement=get_measurement,
        extract_metrics=metric_function,
        initial_coordinates=INITIAL_COORDINATES,
        initial_repeats=INITIAL_REPEATS,
        max_measurements=MAX_MEASUREMENTS,
        progress_callback=show_progress,
        verbose=True,
    )

    print("Optimizer result:", result.coordinate)
    print("Minimum metric:", result.value)