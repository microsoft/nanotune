# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import json
import copy
import time
import logging
import numpy as np
from math import floor
from string import Template
from typing import (
    Optional, Tuple, List, Dict, Any, Sequence, Union, Generator,
    Callable,
)
from contextlib import contextmanager
import matplotlib.pyplot as plt

import qcodes as qc
from qcodes.dataset.experiment_container import load_by_id
from qcodes.utils.helpers import NumpyJSONEncoder

import nanotune as nt
from nanotune.classification.classifier import Classifier
from nanotune.fit.datafit import DataFit
from nanotune.device.gate import Gate
from .take_data import take_data, ramp_to_setpoint
logger = logging.getLogger(__name__)


def save_classification_result(
    run_id: int,
    result_type: Union[str, int],
    result: Union[bool, int],
    meta_tag: Optional[str] = nt.meta_tag,
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
    nt_meta = json.loads(ds.get_metadata(meta_tag))

    if not result_type.startswith("predicted"):
        result_type = "predicted_" + result_type
    nt_meta[result_type] = result
    ds.add_metadata(meta_tag, json.dumps(nt_meta))


def check_measurement_quality(
    classifier: Classifier,
    run_id: int,
    db_name: str,
    db_folder: str,
) -> bool:
    """Apply supplied classifer to determine a measurement's quality.

    Args:
        classifier: Pretrained classifier to use for quality prediction.
        run_id: QCoDeS data run ID.
        db_name: Name of database where dataset is saved.
        db_folder: Path to folder containing database db_name.

    Returns:
        bool: Predicted measurement quality.
    """
    quality = classifier.predict(run_id, db_name, db_folder)
    return any(quality)


def save_extracted_features(
    fit_class: DataFit,
    run_id: int,
    db_name: str,
    db_folder: Optional[str],
    fit_options: Optional[Dict[str, Any]] = None,
) -> None:
    """Performs a data fit and saves extracted features into metadata of the
    QCoDeS dataset.

    Args:
        fit_class:
        run_id: QCoDeS data run ID.
        db_name: Database name where the dataset in question is located.
        db_folder: Path to folder containing database db_name.
    """

    if fit_options is None:
        fit_options = {}

    fit = fit_class(
        run_id,
        db_name,
        **fit_options,
        db_folder=db_folder,
    )
    fit.find_fit()
    fit.save_features()


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
    gates_to_sweep: List[Gate],
    setpoints: List[List[float]],
) -> Generator[None, None, None]:
    """Context manager setting up nanotune gates for a measurement. It ramps the
    gates to their first setpoint before deactivating ramping and yielding a
    generator. At the end, typically after a measurement, ramping is activated
    again.
    Set a post_delay optionally, to make sure the electron gas is settled before
    taking a data point.

    Args:
        gates_to_sweep: Gates to sweep in measurement.
        setpoints: Measurement setpoints

    Returns:
        generator yielding None
    """

    for ig, gate in enumerate(gates_to_sweep):
        gate.dc_voltage(setpoints[ig][0])
        gate.use_ramp(False)
    try:
        yield
    finally:
        for ig, gate in enumerate(gates_to_sweep):
            gate.use_ramp(True)


def set_gate_post_delay(
    gates_to_sweep: List[Gate],
    post_delay: Union[float, List[float]],
) -> None:
    """
    Set it before a measurement to ensure electron settled before taking a
    measurement point.

    Args:
        gates_to_sweep: Gates to sweep in measurement.
        post_delay
    """
    if isinstance(post_delay, float):
        post_delay = len(gates_to_sweep) * [post_delay]
    else:
        assert len(post_delay) == len(gates_to_sweep)

    for ig, gate in enumerate(gates_to_sweep):
        gate.post_delay(post_delay[ig])


def swap_range_limits_if_needed(
    gates_to_sweep: List[Gate],
    current_ranges: List[Tuple[float]],
) -> List[Tuple[float]]:
    """
    Order of current_ranges corresponds to order of gates_to_sweep.
    swaps limits such that starting point will be closest to current voltage.
    To save time and avoid unecessary ramping.

    Args:
        gates_to_sweep: Gates to sweep in measurement.
        current_ranges: Current voltages ranges to sweep. The order in which
            ranges appear in the list is the as in gates_to_sweep.
    """

    new_ranges = copy.deepcopy(current_ranges)
    for gate_idx, c_range in enumerate(current_ranges):
        diff1 = abs(c_range[1] - gates_to_sweep[gate_idx].dc_voltage())
        diff2 = abs(c_range[0] - gates_to_sweep[gate_idx].dc_voltage())

        if diff1 < diff2:
            new_ranges[gate_idx] = (c_range[1], c_range[0])

    return new_ranges


def compute_linear_setpoints(
    ranges: List[Tuple[float, float]],
    voltage_precision: float,
) -> List[List[float]]:
    """Computes linear setpoints the number of points we based on a
    voltage_precision as opposed to a fixed number of points. Useful to ensure
    a minimum resolution required for ML purposes.

    Args:
        ranges: Voltage ranges for all gates to sweep.

    Returns:
        list: Linearly spaced setpoints.
    """

    setpoints_all = []
    for gg, c_range in enumerate(ranges):
        delta = abs(c_range[1] - c_range[0])
        n_points = int(floor(delta / voltage_precision))
        setpoints = np.linspace(c_range[0], c_range[1], n_points)
        setpoints_all.append(setpoints)
    return setpoints_all


def prepare_metadata(
    device_name: str,
    normalization_constants: Dict[str, Tuple[float, float]],
    readout_methods: Dict[str, qc.Parameter],
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
    readout_dict = {k:param.full_name for (k, param) in readout_methods.items()}
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

    for key, value in additional_metadata.items():
        try:
            dump = json.dumps(value, cls=NumpyJSONEncoder)
        except (TypeError, OverflowError) as e:
            raise TypeError(
                f'Adding non-serializable value to meta dict: {value}.'
            )
        meta_dict[key] = value


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
    format_template: Template = '$hours h $minutes min $seconds s',
) -> Tuple[float, str]:
    """Returns the elapsed time in seconds and as a formatted string ready to be
    logged/printed.

    Args:
        start_time:
        end_time:
        format_template: A string Template

    Returns:
        float: Elapsed time in seconds,
        str: Formatted string indicating the elapsed time in hours, minutes and
            seconds.
    """

    elapsed_time = round(float(end_time - start_time), 2)
    hours, minutes = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(minutes, 60)

    formatted_time = Template(format_template).substitute(
        hours=str(hours),
        minutes=str(minutes),
        seconds=str(seconds))

    return elapsed_time, formatted_time


def plot_fit(
    fit_class: DataFit,
    run_id: str,
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
    stage: str,
    measurement_quality: bool,
    predicted_regime: str,
    range_update_directives: List[str],
    termination_reasons: List[str],
) -> None:
    """Prints a tuningstage status on info level of a python logger.

    Args:
        stage: Type of stage.
        measurement_quality: Predicted quality of the last measurement.
        predicted_regime: Predicted quality of the last measurement.
        range_update_directions: Directives to update voltage ranges of
            gates swept during current tuning stage.
        termination_reasons: Potential reasons why the current tuning stage is
            going to stop.
    """

    qual = 'good' if measurement_quality else 'poor'
    msg = (
        stage + ': ' + qual + ' result measured.' + '\n',
        'predicted regime: ' + predicted_regime + '\n',
        'voltage range updates to do: ' + ', '.join(range_update_directives),
        '\n',
        'termination reasons: ' + ', '.join(termination_reasons) + '\n'
    )
    logger.info(msg)


def take_data_add_metadata(
    gates_to_sweep: List[Gate],
    parameters_to_measure: List[qc.Parameter],
    setpoints: List[List[float]],
    pre_measurement_metadata: Dict[str, Any],
    finish_early_check: Optional[Callable[[Dict[str, float]], bool]] = None,
    do_at_inner_setpoint: Optional[Callable[[Any], None]] = None,
    meta_tag: Optional[str] = nt.meta_tag,
) -> int:
    """
    Args:
        gates_to_sweep: List of nt.Gate to sweep.
        parameters_to_measure: List of qc.Parameters to read out.
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
    with set_up_gates_for_measurement(gates_to_sweep, setpoints):
        run_id, n_measured = take_data(
            [gate.dc_voltage for gate in gates_to_sweep],
            parameters_to_measure,
            setpoints,
            finish_early_check=finish_early_check,
            do_at_inner_setpoint=do_at_inner_setpoint,
            metadata_addon=(meta_tag, pre_measurement_metadata)
        )
    seconds, formatted_str = get_elapsed_time(start_time, time.time())
    logger.info('Elapsed time to take data: ' + formatted_str)

    additional_metadata = {
        'n_points': n_measured,
        'elapsed_time': seconds,
    }
    save_metadata(
        run_id,
        additional_metadata,
        meta_tag,
    )

    return run_id





