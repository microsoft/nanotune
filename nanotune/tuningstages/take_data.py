import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence

import numpy as np
import qcodes as qc
from qcodes.dataset.measurements import Measurement

logger = logging.getLogger(__name__)


def take_data(
    parameters_to_sweep: List[qc.Parameter],
    parameters_to_measure: List[qc.Parameter],
    setpoints: Sequence[Sequence[float]],
    finish_early_check: Optional[Callable[[Dict[str, float]], bool]] = None,
    do_at_inner_setpoint: Optional[Callable[[Any], None]] = None,
    metadata_addon: Optional[Tuple[str, Dict[str, Any]]] = None,
) -> int:
    """
    Take 1D or 2D measurements with QCoDeS.

    Args:
        parameters_to_sweep:
        parameters_to_measure:
        setpoints:
        finish_early_check:
        metadata_addon: string with tag under which it will be added
            and metadata iself. To save some important metadata before
            measuring.

    Returns:
        int:
        list: List of the number of points measured in each dimension. If the
            measurement has been stopped early, these will differ from the
            number of setpoints.
    """
    if do_at_inner_setpoint is None:
        do_at_inner_setpoint = do_nothing
    if finish_early_check is None:
        finish_early_check = lambda output: False

    meas = Measurement()
    output = []
    output_dict: Dict[str, float] = {}

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

            if len(parameters_to_sweep) == 2:
                start_voltage = setpoints[1][0]
                do_at_inner_setpoint((parameters_to_sweep[1], start_voltage))
                parameters_to_sweep[1](start_voltage)

                for set_point1 in setpoints[1]:
                    parameters_to_sweep[1](set_point1)
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
                        *output,  # type: ignore
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
                datasaver.add_result((paramx, set_point0), *output)  # type: ignore
                done = finish_early_check(output_dict)
            if done:
                break

    return datasaver.run_id


def do_nothing(param_setpoint_input: Tuple[qc.Parameter, float]) -> None:
    """Default function used at inner setpoints in ```take_data```, not doing
    anything.
    """



def ramp_to_setpoint(param_setpoint_input: Tuple[qc.Parameter, float]) -> None:
    """Ramps nanotune gates (or other instrument parameter with 'use_ramp'
    attribute)to a new setpoint. Sets `use_ramp` back to false after ramping.

    Args:
        param_setpoint_input: Tuple of the QCoDeS parameter and its new
        setpoint.
    """

    qcodes_parameter = param_setpoint_input[0]
    voltage = param_setpoint_input[1]
    try:
        qcodes_parameter.instrument.use_ramp(True)  # type: ignore
        qcodes_parameter(voltage)
        qcodes_parameter.instrument.use_ramp(False)  # type: ignore
    except AttributeError:
        logger.warning("Unable to ramp to new voltage. It will be set.")
