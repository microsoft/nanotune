# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import json
import copy
import time
import logging
import datetime
import numpy as np
from string import Template
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Any,
    Sequence,
    Union,
    Generator,
    Callable,
    Type,
)
from typing_extensions import TypedDict, Literal
from contextlib import contextmanager
import matplotlib.pyplot as plt
from sqlite3 import OperationalError

import qcodes as qc
from qcodes.dataset.experiment_container import load_by_id
from qcodes.utils.helpers import NumpyJSONEncoder

import nanotune as nt
from nanotune.classification.classifier import Classifier
from nanotune.fit.datafit import DataFit
from .take_data import take_data
from nanotune.device_tuner.tuningresult import TuningResult

SetpointSettingsDict = TypedDict(
    "SetpointSettingsDict",
    {
        "parameters_to_sweep": List[qc.Parameter],
        "current_valid_ranges": List[Tuple[float, float]],
        "safety_voltage_ranges": List[Tuple[float, float]],
        "voltage_precision": float,
    },
)
DataSettingsDict = TypedDict(
    "DataSettingsDict",
    {
        "db_name": str,
        "db_folder": str,
        "normalization_constants": Dict[str, Tuple[float, float]],
        "segment_size": float,
        "segment_db_name": str,
        "segment_db_folder": str,
    },
    total=False,
)
ReadoutMethodsLiteral = Literal["dc_current", "dc_sensor", "rf"]
ReadoutMethodsDict = TypedDict(
    "ReadoutMethodsDict",
    {
        "dc_current": qc.Parameter,
        "dc_sensor": qc.Parameter,
        "rf": qc.Parameter,
    },
    total=False,
)
logger = logging.getLogger(__name__)


def save_machine_learning_result(
    run_id: int,
    ml_result: Dict[str, Any],
    meta_tag: str = nt.meta_tag,
) -> None:
    """Saves a classification result such as quality or regime to metadata.

    Args:
        run_id: QCoDeS data run ID.
        result_type: Specifies which classification result is being saved.
            Currently either quality or regime, saved for example as
            'predicted_quality' or 'predicted_regime'.
        result: The classification result.
        meta_tag: Tag under which metadata is saved. Used in QCoDeS'
            dataset.add_metadata method.
    """

    ds = load_by_id(run_id)
    try:
        nt_meta = json.loads(ds.get_metadata(nt.meta_tag))
    except (RuntimeError, TypeError, OperationalError) as r:
        nt_meta = {}

    for result_type, result_value in ml_result.items():
        if not result_type.startswith("predicted"):
            result_type = "predicted_" + result_type
        nt_meta[result_type] = result_value
    ds.add_metadata(meta_tag, json.dumps(nt_meta))


def check_measurement_quality(
    classifier: Classifier,
    run_id: int,
    db_name: Optional[str] = None,
    db_folder: Optional[str] = None,
) -> bool:
    """Applies supplied classifer to determine a measurement's quality.

    Args:
        classifier: Pretrained classifier to use for quality prediction.
        run_id: QCoDeS data run ID.
        db_name: Name of database where dataset is saved.
        db_folder: Path to folder containing database db_name.

    Returns:
        bool: Predicted measurement quality.
    """
    if db_name is None:
        db_name, db_folder = nt.get_database()
    elif db_folder is None:
        _, db_folder = nt.get_database()

    quality = classifier.predict(run_id, db_name, db_folder)
    return any(quality)


def save_extracted_features(
    fit_class: Type[DataFit],
    run_id: int,
    db_name: str,
    db_folder: Optional[str],
) -> None:
    """Performs a data fit and saves extracted features into metadata of the
    QCoDeS dataset.

    Args:
        fit_class:
        run_id: QCoDeS data run ID.
        db_name: Database name where the dataset in question is located.
        db_folder: Path to folder containing database db_name.
    """

    fit = fit_class(
        run_id,
        db_name,
        db_folder=db_folder,
    )
    fit.find_fit()
    fit.save_features()


def get_extracted_features(
    fit_class: Type[DataFit],
    run_id: int,
    db_name: str,
    db_folder: Optional[str],
) -> Dict[str, Any]:
    """Performs a data fit and returns the extracted features.

    Args:
        fit_class:
        run_id: QCoDeS data run ID.
        db_name: Database name where the dataset in question is located.
        db_folder: Path to folder containing database db_name.

    Returns:
        dict: Extracted features.
    """

    fit = fit_class(
        run_id,
        db_name,
        db_folder=db_folder,
    )
    fit.find_fit()
    return fit.features


def get_measurement_features(
    run_id: int,
    db_name: str,
    db_folder: Optional[str],
) -> Dict[str, Any]:
    """Loads data into a nanotune Dataset and returns features previously saved
    to metadata.

    Args:
        run_id: QCoDeS data run ID.
        db_name: Database name where the dataset in question is located.
        db_folder: Path to folder containing database db_name.

    Return:
        dict: Features
    """

    ds = nt.Dataset(run_id, db_name, db_folder=db_folder)
    return ds.features


@contextmanager
def set_up_gates_for_measurement(
    parameters_to_sweep: List[qc.Parameter],
    setpoints: List[List[float]],
) -> Generator[None, None, None]:
    """Context manager setting up gates before a measurement. It ramps them
    to their first setpoint before deactivating ramping (if supported) and
    yields a generator. At the end, typically after a measurement, ramping is
    activated again. If the input parameters do not belong to a nanotune Gate,
    ramping is not deactivated.

    Args:
        parameters_to_sweep: Voltage parameters of gates which are to be swept
        in the measurement.
        setpoints: Measurement setpoints.

    Returns:
        generator yielding None
    """

    for idx, param in enumerate(parameters_to_sweep):
        param(setpoints[idx][0])
        if hasattr(param.instrument, "use_ramp"):
            param.instrument.use_ramp(False)  # type: ignore
        else:
            logger.info("Not turning off ramping when setting up gates.")
    try:
        yield
    finally:
        for idx, param in enumerate(parameters_to_sweep):
            if hasattr(param.instrument, "use_ramp"):
                param.instrument.use_ramp(True)  # type: ignore


def set_post_delay(
    parameters: List[qc.Parameter],
    post_delay: Union[float, List[float]],
) -> None:
    """
    Set qc.Parameter post delay. Can be used before a measurement to set a
    waiting time after a voltage has been set to ensure the electron gas
    settles before taking a measurement point.

    Args:
        parameters: List of QCoDeS parameters.
        post_delay: Post delay to set. Single value input if all parameters
            should have the set post_delay, list of values if they require
            different values.
    """

    if isinstance(post_delay, float):
        post_delay = len(parameters) * [post_delay]
    else:
        assert len(post_delay) == len(parameters)

    for ig, param in enumerate(parameters):
        param.post_delay = post_delay[ig]


def swap_range_limits_if_needed(
    current_voltages: List[float],
    current_valid_ranges: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Saw start and end points of a sweep depending on the current voltages set
    on gates. To save time and avoid unnecessary ramping.
    Order of current_voltages and current_valid_ranges has to match, i.e. the
    voltage value and range at a particular index come from the same voltage
    parameter/gate.

    Args:
        current_voltages: List of voltages currently set to the voltage
            parameters/gates of interest.
        current_valid_ranges: Current voltages ranges to sweep.

    Returns:
        list: Voltage ranges to sweep.
    """

    new_ranges = copy.deepcopy(current_valid_ranges)
    for idx, c_range in enumerate(current_valid_ranges):
        diff1 = abs(c_range[1] - current_voltages[idx])
        diff2 = abs(c_range[0] - current_voltages[idx])

        if diff1 < diff2:
            new_ranges[idx] = (c_range[1], c_range[0])

    return new_ranges


def compute_linear_setpoints(
    ranges: List[Tuple[float, float]],
    voltage_precision: float,
) -> List[List[float]]:
    """Computes linear setpoints the number of points we based on a
    voltage_precision as opposed to a fixed number of points. Useful to ensure
    a minimum resolution required for ML purposes.

    Args:
        ranges: Voltage ranges for all voltage parameters/gates to sweep.

    Returns:
        list: Linearly spaced setpoints.
    """

    setpoints_all = []
    for c_range in ranges:
        delta = abs(c_range[1] - c_range[0])
        n_points = int(round(delta / voltage_precision))
        setpoints = np.linspace(c_range[0], c_range[1], n_points)
        setpoints_all.append(setpoints)
    return setpoints_all


def prepare_metadata(
    device_name: str,
    normalization_constants: Dict[str, Tuple[float, float]],
    readout_methods: ReadoutMethodsDict,
) -> Dict[str, Any]:
    """Sets up a metadata dictionary with fields known prior to a measurement
    set.

    Args:
        normalization_constants: Normalization constants
        readout_methods: Dictionary with readout parameters.

    Returns:
        dict: Metadata dict with fields known prior to a measurement filled in.
    """
    nt_meta = dict.fromkeys(nt.config["core"]["meta_fields"])

    nt_meta["normalization_constants"] = normalization_constants
    nt_meta["git_hash"] = nt.git_hash
    nt_meta["device_name"] = device_name
    readout_dict = {k: param.full_name for (k, param) in readout_methods.items()}  # type: ignore
    nt_meta["readout_methods"] = readout_dict
    nt_meta["features"] = {}

    return nt_meta


def add_metadata_to_dict(
    meta_dict: Dict[str, Any],
    additional_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Adds metadata to an existing dict. Checks if content is serializible by
    QCoDeS.

    Args:
        meta_dict: Existing metadata dictionary.
        additional_metadata: Additional key-value items to add to existing
            metadta dictionary.

    Returns:
        dict: New metadata dictionary.
    """

    new_meta_dict = copy.deepcopy(meta_dict)
    for key, value in additional_metadata.items():
        try:
            dump = json.dumps(value, cls=NumpyJSONEncoder)
        except (TypeError, OverflowError) as e:
            raise TypeError(f"Adding non-serializable value to meta dict: {value}.")
        new_meta_dict[key] = value

    return new_meta_dict


def save_metadata(
    run_id: int,
    meta_dict: Dict[str, Any],
    meta_tag: str,
) -> None:
    """Adds metadata to a QCoDeS dataset.

    Args:
        meta_dict: Dictionary to be added to metadata of a QCoDeS dataset.
        meta_tag: Tag under which the metadata will be stored, i.e. the tag used
            in qc.dataset.add_metadata().
    """

    ds = load_by_id(run_id)
    metadata = json.loads(ds.get_metadata(meta_tag))
    metadata.update(meta_dict)
    ds.add_metadata(meta_tag, json.dumps(metadata, cls=NumpyJSONEncoder))


def get_elapsed_time(
    start_time: float,
    end_time: float,
    format_template: Template = Template("$hours h $minutes min $seconds s"),
) -> Tuple[float, str]:
    """Returns the elapsed time in seconds and as a formatted string ready to be
    logged/printed.

    Args:
        start_time: Start time of event which is being timed.
        end_time: Time at which the timed event finished.
        format_template: A string Template which needs to contain '$hours',
            '$minutes' and $'seconds' as substrings.

    Returns:
        float: Elapsed time in seconds,
        str: Formatted string indicating the elapsed time in hours, minutes and
            seconds.
    """

    elapsed_time = round(float(end_time - start_time), 2)
    hours, minutes = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(minutes, 60)

    formatted_time = format_template.substitute(
        hours=str(hours), minutes=str(minutes), seconds=str(seconds)
    )

    return elapsed_time, formatted_time


def plot_fit(
    fit_class: Type[DataFit],
    run_id: int,
    db_name: str,
    db_folder: Optional[str] = None,
) -> None:
    """Plots a data fit.

    Args:
        fit_class: The DataFit subclass to be used.
        run_id: QCoDeS data run ID.
        db_name: Database name where the dataset in question is located.
        db_folder: Path to folder containing database db_name.
    """

    df = fit_class(run_id, db_name, db_folder=db_folder)
    df.plot_fit()
    plt.show()


def print_tuningstage_status(
    tuning_result: TuningResult,
) -> None:
    """Prints a tuningstage status on info level of a python logger.

    Args:
        tuning_result: TuningResult instance.
    """
    quality = "Good" if tuning_result.success else "Poor"
    if not tuning_result.termination_reasons:
        t_reasons = "None"
    else:
        t_reasons = ", ".join(tuning_result.termination_reasons)

    msg = (
        f"{tuning_result.stage}: {quality} result measured. "
        f"Regime: {tuning_result.ml_result['regime']}. "
        "Termination reasons: " + t_reasons + ". "
    )
    logger.info(msg)
    print(msg)


def take_data_add_metadata(
    parameters_to_sweep: List[qc.Parameter],
    parameters_to_measure: List[qc.Parameter],
    setpoints: List[List[float]],
    pre_measurement_metadata: Dict[str, Any],
    finish_early_check: Optional[Callable[[Dict[str, float]], bool]] = None,
    do_at_inner_setpoint: Optional[Callable[[Any], None]] = None,
    meta_tag: str = nt.meta_tag,
) -> int:
    """Takes 1D or 2D data and saves relevant metadata to the dataset.

    Args:
        parameters_to_sweep: List of qc.Parameters to sweep.
        parameters_to_measure: List of qc.Parameter to read out.
        setpoints: Voltage setpoints to measure.
        pre_measurement_metadata: Metadata dictionary to be saved before a
            measurement starts.
        finish_early_check: Function to be called to check if a measurement
            can be stopped early.
        do_at_inner_setpoint: Function to be called before a new iteration of
            the inner for-loop resumes. Example: Sweep a gate to a new value if
            ramping is turned off.
        Tag under which the metadata will be stored, i.e. the tag used
            in qc.dataset.add_metadata().

    Returns:
        int: QCoDeS data run ID.
    """

    start_time = time.time()
    with set_up_gates_for_measurement(parameters_to_sweep, setpoints):
        run_id, n_measured = take_data(
            parameters_to_sweep,
            parameters_to_measure,
            setpoints,
            finish_early_check=finish_early_check,
            do_at_inner_setpoint=do_at_inner_setpoint,
            metadata_addon=(meta_tag, pre_measurement_metadata),
        )
    seconds, formatted_str = get_elapsed_time(start_time, time.time())
    logger.info("Elapsed time to take data: %s", formatted_str)

    additional_metadata = {
        "n_points": n_measured,
        "elapsed_time": seconds,
    }
    save_metadata(
        run_id,
        additional_metadata,
        meta_tag,
    )

    return run_id


def run_stage(
    stage: str,
    voltage_ranges: List[Tuple[float, float]],
    compute_setpoint_task: Callable[
        [List[Tuple[float, float]]], Sequence[Sequence[float]]
    ],
    measure_task: Callable[[Sequence[Sequence[float]]], int],
    machine_learning_task: Callable[[int], Any],
    save_machine_learning_result: Callable[[int, Any], None],
    validate_result: Callable[[Any], bool],
) -> TuningResult:
    """Executes basic tasks of a tuning stage using functions supplied as input:
        - computes setpoints
        - perform the actual measurement, i.e. take data
        - perform a machine learning task, e.g. classification
        - validate the machine learning result, e.g. check if a good regime was
            found
        - collect all information in a TuningResult instance.
    It does not set back voltages to initial values.

    Args:
        stage: Name/indentifier of the tuning stage.
        voltage_ranges: List of voltages ranges to sweep.
        compute_setpoint_task: Function computing setpoints.
        measure_task: Functions taking data.
        machine_learning_task: Function performing the required machine learning
            task.
        save_machine_learning_result: Function saving machine learning result.
            E.g. save prediction to metadata of the dataset.
        validate_result: Function validating the machine learning
            result/prediction.

    Returns:
        TuningResult: Currently without db_name and db_folder set.
    """

    current_setpoints = compute_setpoint_task(voltage_ranges)
    current_id = measure_task(current_setpoints)

    ml_result = machine_learning_task(current_id)
    save_machine_learning_result(current_id, ml_result)
    success = validate_result(ml_result)

    tuning_result = TuningResult(
        stage,
        success,
        termination_reasons=[],
        data_ids=[current_id],
        ml_result=ml_result,
        timestamp=datetime.datetime.now().isoformat(),
    )

    return tuning_result


def iterate_stage(
    stage: str,
    current_valid_ranges: List[Tuple[float, float]],
    safety_voltage_ranges: List[Tuple[float, float]],
    run_stage: Callable[
        [
            str,
            List[Tuple[float, float]],
            Callable[[List[Tuple[float, float]]], Sequence[Sequence[float]]],
            Callable[[Sequence[Sequence[float]]], int],
            Callable[[int], Any],
            Callable[[int, Any], None],
            Callable[[Any], bool],
        ],
        TuningResult,
    ],
    run_stage_tasks: Tuple[
        Callable[[List[Tuple[float, float]]], Sequence[Sequence[float]]],
        Callable[[Sequence[Sequence[float]]], int],
        Callable[[int], Any],
        Callable[[int, Any], None],
        Callable[[Any], bool],
    ],
    conclude_iteration: Callable[
        [
            TuningResult,
            List[Tuple[float, float]],
            List[Tuple[float, float]],
            int,
            int,
        ],
        Tuple[bool, List[Tuple[float, float]], List[str]],
    ],
    display_result: Callable[[int, TuningResult], None],
    max_n_iterations: int,
) -> TuningResult:
    """Performs several iterations of a run_stage function, a sequence of basic
    tasks of a tuning stage. If desired, and implemented in conclude_iteration,
    new voltage ranges to sweep are determined for the iteration. Issues
    encountered are saved in the TuningStage instance under termination_reasons.
    It does not set back voltages to initial values.

    Args:
        stage: Name/indentifier of the tuning stage.
        current_valid_ranges: List of voltages ranges to sweep.
        run_stage: Function executing the sequence of steps of a tuning stage.
        run_stage_tasks: All input functions of run_stage.
        conclude_iteration: Function checking the outcome of an iteration and
            possibly adjusting voltage ranges if needed. Returns a list of
            termination reasons if the current iteration is to be abandoned.
        display_result: Function to show result of the current iteration.
        max_n_iterations: Maximum number of iterations to perform abandoning.

    Returns:
        TuningResult: Tuning results of the last iteration, with the dataids
            field containing QCoDeS run IDs of all datasets measured.
    """

    done = False
    current_iteration = 0
    run_ids = []

    while not done:
        current_iteration += 1
        tuning_result = run_stage(stage, current_valid_ranges, *run_stage_tasks)
        run_ids += tuning_result.data_ids

        done, current_valid_ranges, termination_reasons = conclude_iteration(
            tuning_result,
            current_valid_ranges,
            safety_voltage_ranges,
            current_iteration,
            max_n_iterations,
        )
        tuning_result.termination_reasons = termination_reasons

        display_result(tuning_result.data_ids[-1], tuning_result)

    tuning_result.data_ids = sorted(list(set(run_ids)))

    return tuning_result


def conclude_iteration_with_range_update(
    tuning_result: TuningResult,
    current_valid_ranges: List[Tuple[float, float]],
    safety_voltage_ranges: List[Tuple[float, float]],
    get_range_update_directives: Callable[
        [int, List[Tuple[float, float]], List[Tuple[float, float]]],
        Tuple[List[str], List[str]],
    ],
    get_new_current_ranges: Callable[
        [List[Tuple[float, float]], List[Tuple[float, float]], List[str]],
        List[Tuple[float, float]],
    ],
    current_iteration: int,
    max_n_iterations: int,
) -> Tuple[bool, List[Tuple[float, float]], List[str]]:
    """Implements a conclude_iteration function for iterate_stage, which
    determines new voltage ranges if the last measurement was not successful.

    Args:
        tuning_result: Tuning result of current run_stage iteration.
        current_valid_ranges: List of the last voltage ranges swep.
        safety_voltage_ranges: List of safety voltages for each voltage
            parameter swept.
        get_range_update_directives: Function to compile a list of directives
            indicating how voltages need to be changed.
        get_new_current_ranges: Function applying list of range change
            directives and returning new voltage ranges.
        current_iteration: Current iteration number.
        max_n_iterations: Maximum number of tuning stage runs to perform.

    """

    new_voltage_ranges: List[Tuple[float, float]] = []
    success = tuning_result.success
    if success:
        done = True
        termination_reasons: List[str] = []
    else:
        (range_update_directives,
         termination_reasons) = get_range_update_directives(
            tuning_result.data_ids[-1],
            current_valid_ranges,
            safety_voltage_ranges,
        )

        if not range_update_directives:
            done = True
        else:
            new_voltage_ranges = get_new_current_ranges(
                current_valid_ranges, safety_voltage_ranges, range_update_directives
            )
            done = False

    if current_iteration >= max_n_iterations:
        done = True
        termination_reasons.append("max current_iteration reached")

    return done, new_voltage_ranges, termination_reasons


def get_current_voltages(
    parameters: List[qc.Parameter],
) -> List[float]:
    """Returns the values set to parameters in ``parameters``.

    Args:
        parameters: List of QCoDeS parameters, i.e. voltage parameters of
            gates.

    Returns:
        list: Values set the input parameters are at, in the same order as
        parameters in ``parameters``.
    """

    return [param() for param in parameters]  # type: ignore


def set_voltages(
    parameters: List[qc.Parameter],
    voltages_to_set: List[float],
) -> None:
    """Set voltages in ``voltages_to_set`` to voltage parameters in
    ``parameters``.

    Args:
        parameters: List of QCoDeS parameters, i.e. voltage parameters of gates.
        voltages_to_set: List of voltages, in the same order as parameters in
            ``parameters``.
    """

    [param(value) for param, value in zip(parameters, voltages_to_set)]


def get_fit_range_update_directives(
    fit_class: Type[DataFit],
    run_id: int,
    db_name: str,
    db_folder: Optional[str],
) -> List[str]:
    """Returns voltage range update directives determined from a fit.

    Args:
        fit_class: Data fit class to use for fitting.
        run_id: QCoDeS data run ID.
        db_name: Database name.
        db_folder: Path to folder where database is saved.

    Returns:
        list: List of strings indicating in which direction voltage ranges of
            the gates swept need to be changed.
    """

    fit = fit_class(
        run_id,
        db_name,
        db_folder=db_folder,
    )
    return fit.range_update_directives
