"""Run a Gaussian-process beam-steering search using MuSR asymmetry data.

In normal operation, measurements are acquired through Genie.  When
TEST_MODE is enabled, the acquisition callback selects the nearest
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

TEST_MODE = True
PROCESSED_RUNS_LOG = "processed_runs.log"
CALCULATE_ALPHA = False
SUPPRESS_MANTID_STATUS_LOGS = True
LONGITUDINAL_MODE = True

SCAN_PVS = [
    {"name": "hsm_steering", "pv": "IN:MUSR:KEPCO_04:CURRENT:SP", "bounds": (-0.5, 0.5), "precision": 2, "minimum_step_size": 0.01},
    {"name": "vsm_steering", "pv": "IN:MUSR:KEPCO_05:CURRENT:SP", "bounds": (-1.5, 1.5), "precision": 2, "minimum_step_size": 0.01},
]

INITIAL_COORDINATES = [(-0.20,  0.20), (-0.20, -0.20), (0.20, 0.20), (0.20, -0.20), (0.00, 0.00), (0.10, -0.60)]
MEVENTS_PER_MEASUREMENT = 1
SETTLE_SECONDS = 2.0
ACCURACY = 0.0001
INITIAL_REPEATS = 2
MAX_MEASUREMENTS = 10
LAST_GOOD_VALUE = 14


# ==========================================================================
# SETUP & CONFIGURATION PARSING
# ==========================================================================

if TEST_MODE:
    INPUT_DIR = Path("data/incoming_nexus")
    OUTPUT_DIR = Path("data/processed_nexus")
    ALPHA_FILE = None
    DUMMY_NEXUS_DIR = Path("data/dummy_nexus")
    dummy_run_files = sorted(Path(DUMMY_NEXUS_DIR).glob("musr*.nxs"))
    if not dummy_run_files:
        raise FileNotFoundError(f"No dummy Nexus files found in {DUMMY_NEXUS_DIR}.")
    
    first_dummy_run = re.fullmatch(r"musr(\d+)\.nxs", dummy_run_files[0].name, re.IGNORECASE)
    if first_dummy_run is None:
        raise ValueError(f"First dummy Nexus file has an invalid run name: {dummy_run_files[0].name}.")
    START_RUN_NUMBER = int(first_dummy_run.group(1).lstrip("0") or "0")

else:
    INPUT_DIR = Path("//musr/Data")
    OUTPUT_DIR = Path("data/processed_nexus")
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

SPECTRA_CFG = {
    True: {"forward": list(range(33, 65)), "backward": list(range(1, 33)), "first": 0, "second": 1},
    False: {"forward": list(range(9, 17)) + list(range(57, 65)), "backward": list(range(25, 33)) + list(range(41, 49)), "first": 3, "second": 1}
}[LONGITUDINAL_MODE]


# ==========================================================================
# PATH & RUN LOGIC
# ==========================================================================

def get_run_from_path(file_path: Path) -> int:
    """
    Extract the integer run number from a file path.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the file (e.g., 'musr0001234.txt' or 'processed_musr001.nxs').

    Returns
    -------
    int
        The extracted integer run number.
    """
    clean_name = file_path.stem.replace("processed_", "").replace("musr", "").replace("dummy_", "")
    return int(clean_name)

def get_path_from_run(directory: Path, run_number: int, extension: str, prefix: str = "musr") -> Path:
    """
    Generate a consistently formatted file path with zero-padded run numbers.

    Parameters
    ----------
    directory : pathlib.Path
        The directory where the file will reside.
    run_number : int
        The integer run number.
    extension : str
        The file extension, including the dot (e.g., '.nxs' or '.txt').
    prefix : str, optional
        The prefix before the run number, by default "musr".

    Returns
    -------
    pathlib.Path
        The constructed file path.
    """
    return directory / f"{prefix}{run_number:08d}{extension}"


# ==========================================================================
# STATE ENCAPSULATION
# ==========================================================================

class BeamOptimizerState:
    """
    Encapsulates the state and data reduction logic for the beam optimizer.

    Parameters
    ----------
    input_dir : pathlib.Path
        Directory where incoming raw Nexus files are stored.
    output_dir : pathlib.Path
        Directory where processed text files will be saved.
    start_run_number : int
        The run number marking the beginning of the optimization sequence.
    """

    def __init__(self, input_dir: Path, output_dir: Path, start_run_number: int):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.start_run_number = start_run_number
        
        self.bin_params_cache = None
        self.dummy_measurement_files = None
        self.dummy_measurement_paths = set()
        self.next_test_run_number = 900000

    def sync_and_process_batch(self, measurements: Sequence[Mapping[str, Any]], alpha: Any) -> None:
        """
        Wait for the newest measurement and reduce any unprocessed Nexus files.

        Parameters
        ----------
        measurements : Sequence[Mapping[str, Any]]
            A list of measurement dictionaries containing 'run_number'.
        alpha : Any
            The Mantid detector-pair asymmetry correction.
        """
        expected_run = int(measurements[-1]["run_number"])
        
        while True:
            expected_path = get_path_from_run(self.input_dir, expected_run, ".nxs")
            if expected_path.exists():
                break
            time.sleep(1)

        processed_log_path = self.output_dir / PROCESSED_RUNS_LOG
        processed_runs = self._load_processed_runs(processed_log_path)

        for nxs_path in sorted(self.input_dir.glob("musr*.nxs")):
            run_number = get_run_from_path(nxs_path)
            
            if run_number < self.start_run_number or run_number in processed_runs:
                continue

            ws, params = self.get_asymmetry(nxs_path, alpha)
            
            out_path = get_path_from_run(self.output_dir, run_number, ".txt")
            temp_path = out_path.with_suffix(".tmp")
            
            self._write_asymmetry_to_txt(ws, params, temp_path, nxs_path)
            temp_path.rename(out_path) 

            self._log_processed_run(processed_log_path, run_number)
            processed_runs.add(run_number)

    def _load_processed_runs(self, log_path: Path) -> set[int]:
        """
        Load the set of already processed run numbers from the log file.

        Parameters
        ----------
        log_path : pathlib.Path
            Path to the processed runs log file.

        Returns
        -------
        set[int]
            A set of integer run numbers that have been processed.
        """
        if not log_path.exists():
            return set()
        with log_path.open("r", encoding="utf-8") as f:
            return {int(line.strip()) for line in f if line.strip()}

    def _log_processed_run(self, log_path: Path, run_number: int) -> None:
        """
        Record a successfully processed run number to the log file.

        Parameters
        ----------
        log_path : pathlib.Path
            Path to the processed runs log file.
        run_number : int
            The run number to log.
        """
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{run_number}\n")
            
    def _write_asymmetry_to_txt(self, ws: Any, params: dict[str, float], out_path: Path, nxs_path: Path) -> None:
        """
        Extract time, asymmetry, and errors from a workspace and write to a text file.

        Parameters
        ----------
        ws : Any
            Mantid asymmetry workspace.
        params : dict[str, float]
            Dictionary of steering PV parameters for the file header.
        out_path : pathlib.Path
            Path to write the output text file.
        nxs_path : pathlib.Path
            Path to the source Nexus file (used to detect dummy runs for noise injection).
        """
        time_data, asym_data, error_data = ws.extractX()[0], ws.extractY()[0], ws.extractE()[0]
        
        if nxs_path in self.dummy_measurement_paths:
            asym_data = asym_data + np.random.normal(0.0, error_data, size=asym_data.shape)

        with out_path.open('w', encoding="utf-8") as output_file:
            valid_params = {k: v for k, v in params.items() if v is not None}
            if valid_params:
                header = '_'.join([f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in valid_params.items()])
                output_file.write(f"# {header}\n")
            for t, a, e in zip(time_data, asym_data, error_data):
                output_file.write(f"{t}\t{a}\t{e}\n")

    def get_asymmetry(self, nexus_file_path: Path, alpha: Any) -> tuple[Any, dict[str, float]]:
        """
        Load a single Nexus run and convert it into a paired asymmetry workspace.

        Parameters
        ----------
        nexus_file_path : pathlib.Path
            Path to the raw Nexus run.
        alpha : Any
            Detector-pair asymmetry correction.

        Returns
        -------
        tuple[Any, dict[str, float]]
            The resulting Mantid asymmetry workspace and the extracted steering parameters.
        """
        load_result = LoadMuonNexus(Filename=str(nexus_file_path), OutputWorkspace="test_result", DetectorGroupingTable="grouping", DeadTimeTable='deadtimes')
        loaded_time_zero, loaded_time_good = load_result[2], load_result[3]
        shifted_time_good = loaded_time_good - loaded_time_zero
        shifted_last_good = LAST_GOOD_VALUE - loaded_time_zero
        params = get_parameters(load_result[0])
            
        if self.bin_params_cache is None:
            self.bin_params_cache = calculate_constant_error_bins(
                load_result[0], shifted_time_good, shifted_last_good, 5000, 
                SPECTRA_CFG["forward"], SPECTRA_CFG["backward"], loaded_time_zero
            )

        f_b_counts = MuonProcess(
            InputWorkspace=load_result[0], OutputWorkspace="ws_asym", Mode="Combined",
            DetectorGroupingTable='grouping', SummedPeriodSet="1", LoadedTimeZero=loaded_time_zero,
            OutputType="PairAsymmetry", DeadTimeTable='deadtimes', Xmin=shifted_time_good,
            Xmax=shifted_last_good, Alpha=alpha, ApplyDeadTimeCorrection=True,
            PairFirstIndex=SPECTRA_CFG["first"], PairSecondIndex=SPECTRA_CFG["second"], RebinParams=self.bin_params_cache
        )
        return f_b_counts, params

state = BeamOptimizerState(INPUT_DIR, OUTPUT_DIR, START_RUN_NUMBER)


# ==========================================================================
# MANTID LOGIC
# ==========================================================================

def get_parameters(ws: Any, round_flag: bool = False) -> dict[str, float]:
    """
    Return the configured steering PV averages recorded in a workspace.

    Parameters
    ----------
    ws : Any
        Mantid workspace containing the steering process-variable logs.
    round_flag : bool, optional
        Whether values should be rounded to two decimal places, by default False.

    Returns
    -------
    dict[str, float]
        A mapping from configured PV names to available mean values.
    """
    run = ws.getRun()
    def get_time_avg(log_name: str) -> float | None:
        try:
            return run.getProperty(log_name).getStatistics().time_mean
        except RuntimeError:
            return None
    raw_parameters = {pv["name"]: get_time_avg(pv["name"]) for pv in SCAN_PVS}
    if round_flag:
        raw_parameters = {k: round(v, 2) for k, v in raw_parameters.items()}
    return {k: v for k, v in raw_parameters.items() if v is not None}

def get_alpha(calibration_file: Path, forward_spectra: Sequence[int], backward_spectra: Sequence[int], last_good_value: float) -> Any:
    """
    Calculate the detector-pair asymmetry correction from a calibration run.

    Parameters
    ----------
    calibration_file : pathlib.Path
        Calibration Nexus file path.
    forward_spectra : Sequence[int]
        Forward detector spectrum numbers.
    backward_spectra : Sequence[int]
        Backward detector spectrum numbers.
    last_good_value : float
        End of the valid calibration time range.

    Returns
    -------
    Any
        The Mantid detector-pair asymmetry correction.
    """
    c_ws = LoadMuonNexus(Filename=str(calibration_file), OutputWorkspace="a_ws", DetectorGroupingTable="grouping", DeadTimeTable='deadtimes')
    return AlphaCalc(InputWorkspace='a_ws', ForwardSpectra=forward_spectra, BackwardSpectra=backward_spectra, FirstGoodValue=c_ws[3], LastGoodValue=last_good_value)

def calculate_constant_error_bins(workspace: Any, start_time: float, end_time: float, target_counts: float, forward_spectra: Sequence[int], backward_spectra: Sequence[int], time_zero: float) -> list[float]:
    """
    Build Mantid rebin parameters with approximately constant count errors.

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
    forward_spectra : Sequence[int]
        Forward detector spectrum numbers.
    backward_spectra : Sequence[int]
        Backward detector spectrum numbers.
    time_zero : float
        Time-zero offset to subtract from the X data.

    Returns
    -------
    list[float]
        Mantid-compatible ``[start, step, end, ...]`` rebin parameters.
    """
    x_data = workspace.readX(0) - time_zero
    f_idx = [workspace.getIndexFromSpectrumNumber(s) for s in forward_spectra]
    b_idx = [workspace.getIndexFromSpectrumNumber(s) for s in backward_spectra]
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


# ==========================================================================
# PCA & METRIC EXTRACTION
# ==========================================================================

def get_parameters_from_files(stsp: tuple[int, int], prenum: str, ext: str) -> list[dict[str, float | int]]:
    """
    Read steering metadata written in processed text-file headers.

    Parameters
    ----------
    stsp : tuple[int, int]
        Inclusive first and last run numbers to read.
    prenum : str
        Text prefix before each run number (e.g., 'musr').
    ext : str
        File extension, including its leading dot (e.g., '.txt').

    Returns
    -------
    list[dict[str, float | int]]
        One metadata mapping per run, including its integer run number.
    """
    metadata = []
    pattern = re.compile(r"([a-zA-Z][a-zA-Z_]*)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    for run_number in range(stsp[0], stsp[1] + 1):
        file_path = state.output_dir / f"{prenum}{run_number:08d}{ext}"
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as file:
            title = file.readline().strip()
        matches = pattern.findall(title)
        file_metadata = {"run": run_number}
        for key, val in matches:
            file_metadata[key] = float(val)
        metadata.append(file_metadata)
    return metadata
    
def metric_function(measurements: Sequence[Mapping[str, Any]]) -> list[float]:
    """
    Process all measurements and return one PCA metric per optimizer run.

    Parameters
    ----------
    measurements : Sequence[Mapping[str, Any]]
        Measurement records containing a 'run_number' field.

    Returns
    -------
    list[float]
        One PCA score per measurement, in input order.

    Raises
    ------
    ValueError
        If output data are unavailable or a run is not in PCA output.
    """
    alpha = get_alpha(ALPHA_FILE, SPECTRA_CFG["forward"], SPECTRA_CFG["backward"], LAST_GOOD_VALUE) if CALCULATE_ALPHA else 1
    
    state.sync_and_process_batch(measurements, alpha)

    txt_files = sorted(state.output_dir.glob("musr*.txt"))
    if not txt_files:
        raise ValueError("No text files found in output directory.")
    
    first_run = get_run_from_path(txt_files[0])
    last_run = get_run_from_path(txt_files[-1])

    dh = DataHandler()
    dh.load_batch(loc=str(state.output_dir) + '\\', stsp=(first_run, last_run), prenum="musr00", ext=".txt", skiprows=1)
    
    pvs = get_parameters_from_files((first_run, last_run), "musr", ".txt")
    
    dh.prepare_XYE_PCA()
    pca_machine = PCAMachine(dh)
    pca_machine.perform_pca()
    
    pc1_curve = pca_machine.pc_curves[0][:, 0]
    pc1_errors = np.asarray(dh.prepared_data[0][2], dtype=float)

    valid = np.isfinite(pc1_curve) & np.isfinite(pc1_errors) & (pc1_errors > 0)
    if not np.any(valid):
        raise ValueError("Cannot determine the PC1 sign: no finite prepared bins with positive error bars.")

    weights = 1.0 / np.square(pc1_errors[valid])
    weighted_pc1_sum = np.sum(pc1_curve[valid] * weights)

    if weighted_pc1_sum < 0:
        pca_machine.pc_curves[0][:, 0] *= -1
        pca_machine.pc_scores[0][0, :] *= -1

    pc1_scores = pca_machine.pc_scores[0][0, :].copy()
    score_map = {str(pv_data["run"]): score for pv_data, score in zip(pvs, pc1_scores)}
    
    metrics = []
    for m in measurements:
        run_str = str(m["run_number"]).lstrip("0") or "0"
        if run_str in score_map:
            metrics.append(score_map[run_str])
        else:
            raise ValueError(f"Run {run_str} requested by optimizer but not found in PCA output.")
            
    return metrics


# ==========================================================================
# OPTIMIZER CONTROL
# ==========================================================================

def _get_dummy_measurement_file(coordinate: np.ndarray) -> Path:
    """
    Select and apply the nearest available dummy-run coordinate.

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
    if state.dummy_measurement_files is None:
        state.dummy_measurement_files = []
        for source_path in sorted(DUMMY_NEXUS_DIR.glob("*.nxs")):
            if ALPHA_FILE and source_path == Path(ALPHA_FILE):
                continue
            workspace_name = f"dummy_{source_path.stem}"
            workspace = LoadMuonNexus(Filename=str(source_path), OutputWorkspace=workspace_name, DetectorGroupingTable="grouping", DeadTimeTable="deadtimes")[0]
            parameters = get_parameters(workspace, round_flag=True)
            print(f"Loaded {source_path.name} with parameters: {parameters}")
            state.dummy_measurement_files.append((source_path, parameters))

    parameter_names = [item["name"] for item in SCAN_PVS]
    requested_coordinate = np.asarray(coordinate, dtype=float)
    available_files = [
        (source_path, parameters) for source_path, parameters in state.dummy_measurement_files
        if all(name in parameters for name in parameter_names)
    ]
    if not available_files:
        raise FileNotFoundError("No dummy Nexus files contain all required parameters: " + ", ".join(parameter_names))

    source_path, parameters = min(
        available_files,
        key=lambda item: np.sum((requested_coordinate - np.asarray([item[1][name] for name in parameter_names])) ** 2)
    )
    coordinate[:] = [parameters[name] for name in parameter_names]
    return source_path

def _get_test_destination() -> tuple[Path, str]:
    """
    Return a unique input path and run number for test-mode acquisition.

    Returns
    -------
    tuple[pathlib.Path, str]
        A destination path and its run number as a string.
    """
    while True:
        run_number = state.next_test_run_number
        state.next_test_run_number += 1
        destination_path = get_path_from_run(state.input_dir, run_number, ".nxs")
        archived_path = destination_path.with_name(f"processed_{destination_path.name}")
        if not destination_path.exists() and not archived_path.exists():
            return destination_path, str(run_number)

def get_measurement(coordinate: Sequence[float]) -> dict[str, object]:
    """
    Acquire one measurement through Genie or a matching dummy Nexus run.

    Parameters
    ----------
    coordinate : Sequence[float]
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
        raise ValueError("Coordinate dimension does not match the number of SCAN_PVS.")

    if TEST_MODE:
        source_path = _get_dummy_measurement_file(coordinate)
        destination_path, run_number = _get_test_destination()
        shutil.copy2(source_path, destination_path)
        state.dummy_measurement_paths.add(destination_path)
        return {"run_number": run_number, "coordinate": tuple(float(v) for v in coordinate), "mevents": MEVENTS_PER_MEASUREMENT}

    for item, value in zip(SCAN_PVS, coordinate):
        value = round(value, item.get("precision", 2))
        g.set_pv(item["pv"], float(value), wait=True, is_local=False)
    
    if SETTLE_SECONDS > 0:
        g.waitfor_time(seconds=SETTLE_SECONDS, quiet=True)

    title = ", ".join(f"{item['name']}={float(value):+.6g}" for item, value in zip(SCAN_PVS, coordinate))
    g.change_title(f"GP/KG scan: {title}")
    
    g.begin(quiet=True)
    run_number = str(g.get_runnumber())
    g.waitfor(mevents=MEVENTS_PER_MEASUREMENT, quiet=False)
    actual_mevents = float(g.get_mevents())
    g.end(quiet=True)

    return {"run_number": run_number, "coordinate": tuple(float(v) for v in coordinate), "mevents": actual_mevents}

def show_progress(observations: Sequence[Observation], estimated_noise_std: float | None) -> None:
    """
    Print the current metric and coordinate for every observation.

    Parameters
    ----------
    observations : Sequence[Observation]
        Optimizer observations to display.
    estimated_noise_std : float | None
        Current estimated measurement noise, if available.
    """
    print("\nCurrent global metrics:")
    for index, obs in enumerate(observations):
        vals = ", ".join(f"{item['name']}={value:+.6g}" for item, value in zip(SCAN_PVS, obs.coordinate))
        print(f"  {index:2d}  run={obs.measurement['run_number']}  {vals}  metric={obs.metric:.8g}")

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for entry in OUTPUT_DIR.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    if TEST_MODE and INPUT_DIR is not Path("//musr/Data"):
        for entry in INPUT_DIR.iterdir():
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

    print(f"Optimizer result: {result.coordinate}. The model predicts further scanning would only improve the beam alignment score by {result.kg:.8g}"
          f" compared to predefined accuracy of {ACCURACY}. The noise in this coordinate is estimated to be {result.estimated_noise_std:.8g}"
          f" and the optimizer performed {len(result.observations)} measurements.")
    print(f"Minimum metric: {result.value}")