import json
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
import logging
from math import floor
import qcodes as qc
from qcodes.dataset.measurements import Measurement
import nanotune as nt
logger = logging.getLogger(__name__)

        # current_output_dict: Dict[str, float],
        # readout_methods_to_use: str = 'dc_current',

def take_data(
    parameters_to_sweep: List[qc.Parameter],
    parameters_to_measure: List[qc.Parameter],
    setpoints: List[List[float]],
    finish_early_check: Optional[Callable[[Dict[str, float]], bool]] = None,
    do_at_inner_setpoint: Optional[Callable[[Any], None]] = None,
    metadata_addon: Optional[Tuple[str, Dict[str, Any]]] = None,
) -> Tuple[int, List[int]]:
    """
    Take data for 1 or 2D data
    Args:
        parameters_to_sweep (list):
        parameters_to_measure (list):
        setpoints:
        finish_early_check (callable):
        metadata_addon (tuple): string with tag under which it will be added
            and metadata iself. To save some important metadata before measuring
    """
    if do_at_inner_setpoint is None:
        do_at_inner_setpoint = do_nothing
    if finish_early_check is None:
        finish_early_check = lambda output: False

    meas = Measurement()
    output = []
    output_dict: Dict[str, float] = {}
    n_measured = [0, 0]

    for set_param in parameters_to_sweep:
        meas.register_parameter(set_param)

    for m_param in parameters_to_measure:
        meas.register_parameter(m_param, setpoints=parameters_to_sweep)
        output.append([m_param, None])
        output_dict[m_param.full_name] = np.nan

    done = False

    with meas.run() as datasaver:
        if metadata_addon is not None:
            datasaver.dataset.add_metadata(
                metadata_addon[0], json.dumps(metadata_addon[1])
            )

        for set_point0 in setpoints[0]:
            parameters_to_sweep[0](set_point0)
            n_measured[0] += 1

            if len(parameters_to_sweep) == 2:
                start_voltage = setpoints[1][0]
                do_at_inner_setpoint((parameters_to_sweep[1], start_voltage))
                parameters_to_sweep[1](start_voltage)

                for set_point1 in setpoints[1]:
                    parameters_to_sweep[1](set_point1)
                    n_measured[1] += 1
                    m_params = parameters_to_measure
                    for p, parameter in enumerate(m_params):
                        value = parameter.get()
                        output[p][1] = value
                        output_dict[parameter.full_name] = value

                    paramx = parameters_to_sweep[0].full_name
                    paramy = parameters_to_sweep[1].full_name
                    datasaver.add_result(
                        (paramx, set_point0),
                        (paramy, set_point1),
                        *output, # type: ignore
                    )
                    done = finish_early_check(output_dict)
                    if done:
                        break
            else:
                m_params = parameters_to_measure
                for p, parameter in enumerate(m_params):
                    value = parameter.get()
                    output[p][1] = value
                    output_dict[parameter.full_name] = value

                paramx = parameters_to_sweep[0].full_name
                datasaver.add_result(
                    (paramx, set_point0), *output # type: ignore
                )
                done = finish_early_check(output_dict)
            if done:
                break

    return datasaver.run_id, n_measured

def do_nothing(param_setpoint_input: Tuple[qc.Parameter, float]) -> None:
    pass

def ramp_to_setpoint(param_setpoint_input: Tuple[qc.Parameter, float]) -> None:
    """
    Ramp nanotune gate (or other instrument parameter with 'use_ramp' attribute)
    to new setpoint. Set `use_ramp` back to false
    """
    qcodes_parameter = param_setpoint_input[0]
    voltage = param_setpoint_input[1]
    try:
        qcodes_parameter.instrument.use_ramp(True)  # type: ignore
        qcodes_parameter(voltage)
        qcodes_parameter.instrument.use_ramp(False)  # type: ignore
    except AttributeError as a:
        logger.warning('Unable to ramp to new voltage. It will be set.')

def compute_linear_setpoints(
    ranges: List[Tuple[float, float]],
    voltage_precision: float,
    max_jumps: Optional[List[float]] = None
) -> List[List[float]]:
    """
    # Calculate the number of points we need to cover the entire
        # range without exceeding max_jump if specified
    """
    if max_jumps is None:
        max_jumps = [voltage_precision] * len(ranges)

    setpoints_all = []
    for gg, c_range in enumerate(ranges):
        delta = abs(c_range[1] - c_range[0])
        n_safe = int(floor(delta / max_jumps[gg]))
        n_precise = int(floor(delta / voltage_precision))

        n = np.max([n_safe, n_precise])
        setpoints = np.linspace(c_range[0], c_range[1], n)
        setpoints_all.append(setpoints)
    return setpoints_all
