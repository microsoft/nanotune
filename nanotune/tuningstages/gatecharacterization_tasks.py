# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import logging
import copy
import numpy as np
from typing import (
    Optional, Tuple, List, Dict, Any,
)
from .base_tasks import get_fit_range_update_directives
from nanotune.device.gate import Gate
logger = logging.getLogger(__name__)


def get_new_gatecharacterization_range(
    current_voltage_ranges: List[Tuple[float, float]],
    safety_voltage_ranges: List[Tuple[float, float]],
    range_update_directives: List[str],
) -> List[Tuple[float, float]]:
    """Determines new voltage range for a subsequent tuning stage
    iteration. It extends the currnet range to the relevant safety range, i.e.
    if the directive required a voltage to be swept more negative, then the
    lower bound is set to the lower safety value.
    No intermediate tries, just being efficient.

    Args:
        current_range: Current voltage range.
        safety_range: List of safety ranges.
        range_update_directives: List of range update directives.

    Returns:
        Tuple: New voltage range.
    """

    assert len(current_voltage_ranges) == 1
    assert len(safety_voltage_ranges) == 1

    new_min, new_max = current_voltage_ranges[0]
    safety_range = safety_voltage_ranges[0]

    for directive in range_update_directives:
        if directive not in ["x more negative", "x more positive"]:
            logger.error('Unknown voltage range update directive.')

    if "x more negative" in range_update_directives:
        new_min = safety_range[0]
    if "x more positive" in range_update_directives:
        new_max = safety_range[1]

    return [(new_min, new_max)]


def get_range_directives_gatecharacterization(
    fit_range_update_directives: List[str],
    current_voltage_ranges: List[Tuple[float, float]],
    safety_voltage_ranges: List[Tuple[float, float]],
) -> Tuple[List[str], List[str]]:
    """Determines voltage range directives to update ranges for a subsequent
    tuning stage iteration. It checks if the voltage range update directives
    determined previously, e.g by a fit class, can be carried out based on the
    supplied safety ranges.

    Args:
        fit_range_update_directives: Directives determined previously, such as
            during fitting.
        gate_current_range: Voltage range swept at previous iteration.
        gate_safety_range: Safety range of gate/voltage parameter swept.

    Returns:
        list: Range update directives, e.g. 'x more negative'.
        list: Issues encountered, e.g. 'positive safety voltage reached'.
    """

    safety_v_range = safety_voltage_ranges[0]
    current_v_range = current_voltage_ranges[0]

    neg_range_avail = abs(current_v_range[0] - safety_v_range[0])
    pos_range_avail = abs(current_v_range[1] - safety_v_range[1])

    range_update_directives = []
    issues = []

    if "x more negative" in fit_range_update_directives:
        if neg_range_avail >= 0.1:
            range_update_directives.append("x more negative")
        else:
            issues.append("negative safety voltage reached")

    if "x more positive" in fit_range_update_directives:
        if pos_range_avail >= 0.1:
            range_update_directives.append("x more positive")
        else:
            issues.append("positive safety voltage reached")

    return range_update_directives, issues


def finish_early_pinched_off(
    last_measurement_strength: float,
    normalization_constant: Tuple[float, float],
    recent_measurement_strengths: List[float],
    voltage_precision: float,
    noise_level: float,
    voltage_interval_to_track: float = 0.3,
) -> Tuple[bool, List[float]]:
    """Checks the average strength of measured signal over a given voltage
    interval is below the noise floor. If this is the case, the boolean returned
    indicates that the measurement can be stopped.

    Args:
        last_measurement_strength: Last measurement output.
        normalization_constant: Constant to normalize last_measurement_strength.
        recent_measurement_strengths: List of most recent signal strengths.
        voltage_precision: Voltage precision of the measurement, i.e. voltage
            difference between setpoints.
        noise_level: Relative noise in percent compared to normalised signal.
        voltage_interval_to_track: Voltage interval over which the average
            should be taken. Voltage ranges less than voltage_interval_to_track
            will be ignored, i.e. will always return False.

    Return:
        bool: Whether or not a measurement can be stopped.
        list: List of measurements strenghts/outputs over the last
            voltage_interval_to_track.
    """

    finish = False
    new_recent_output = copy.deepcopy(recent_measurement_strengths)
    new_recent_output.append(last_measurement_strength)
    n_setpoints_to_track = int(voltage_interval_to_track / (voltage_precision))
    if len(new_recent_output) > n_setpoints_to_track:
        new_recent_output = new_recent_output[-n_setpoints_to_track:]
        avg_output = np.mean(new_recent_output)

        n_cts = normalization_constant
        norm_avg = (avg_output - n_cts[0]) / (n_cts[1] - n_cts[0])
        if norm_avg < noise_level:
            finish = True

    return finish, new_recent_output