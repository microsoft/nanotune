import copy
import logging
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping, Sequence,
    MutableMapping, Iterable, Callable
)
from collections import namedtuple
from functools import partial
import numpy as np
import qcodes as qc
from qcodes import validators as vals
from qcodes.instrument.delegate import DelegateInstrument
from qcodes.instrument.delegate.grouped_parameter import (
    DelegateGroup,
    DelegateGroupParameter,
    GroupedParameter,
)
from qcodes.station import Station

import nanotune as nt
from nanotune.device.device_channel import DeviceChannel
from nanotune.device.ohmic import Ohmic
from nanotune.device_tuner.tuningresult import TuningResult


logger = logging.getLogger(__name__)

nrm_cnst_tp = Dict[str, Tuple[float, float]]
vltg_rngs_tp = Dict[int, Tuple[Union[None, float], Union[None, float]]]

def readout_formatter(
    *values: Any,
    param_names: List[str],
    name: str,
    **kwargs: Any,
) -> Any:
    return namedtuple(name, param_names)(*values, **kwargs)


class Device(DelegateInstrument):
    """
    device_type: str, e.g 'fivedot'
    readout_methods = {
        'dc_current': qc.Parameter,
        'dc_sensor': qc.Parameter,
        'rf': qc.Parameter,
    }
    measurement_options = {
        'dc_current': {
            'delay': <float>,
            'inter_delay': <float>,
        }
        'dc_sensor': {
            'delay': <float>,
            'inter_delay': <float>,
        }
        'rf': {
            'delay': <float>,
            'inter_delay': <float>,
        }
    }
    gate_parameters = {
        gate_id: int = {
            'channel_id': int,
            'dac_instrument': DACInterface,
            'label': str,
            'safety_range': Tuple[float, float],
        }
    }
    ohmic_parameters = {
        ohmic_id: int ={
            'dac_instrument': DACInterface,
            'channel_id': int,
            'ohmic_id': int,
            'label': str,

        }
    }
    sensor_parameters = {
        gate_id: int = {
            'channel_id': int,
            'dac_instrument': DACInterface,
            'label': str,
            'safety_range': Tuple[float, float],
        }
    }
    """

    def __init__(
        self,
        name: str,
        station: Station,
        parameters: Optional[Union[Mapping[str, Sequence[str]], Mapping[str, str]]] = None,
        channels: Optional[Union[Mapping[str, Sequence[str]], Mapping[str, str]]] = None,
        readout_parameters: Optional[Mapping[str, str]] = None,
        initial_values: Optional[Mapping[str, Any]] = None,
        set_initial_values_on_load: bool = False,
        device_type: Optional[str] = '',
        initial_valid_ranges: Optional[Mapping[str, Sequence[str]]] = None,
        normalization_constants: Optional[nrm_cnst_tp] = None,
        **kwargs,
    ) -> None:
        print(readout_parameters)

        super().__init__(
            name,
            station,
            parameters,
            channels,
            initial_values,
            set_initial_values_on_load,
            metadata={'device_type': device_type},
            **kwargs,
            )

        self.gates, self.ohmics = self._initialize_channel_lists(channels)

        param_names, paths = list(zip(*list(readout_parameters.items())))
        super()._create_and_add_parameter(
            'readout',
            station,
            paths,
            formatter=partial(
                readout_formatter,
                param_names=param_names,
                name='readout',
            )
        )

        init_valid_ranges_renamed = {}
        if initial_valid_ranges is None:
            for gate in self.gates:
                gate_id = gate.gate_id()
                init_valid_ranges_renamed[gate_id] = gate.safety_range()
        else:
            for gate_name, valid_range in initial_valid_ranges.items():
                gate_id = getattr(self, gate_name).gate_id()
                init_valid_ranges_renamed[gate_id] = valid_range

        self._initial_valid_ranges: vltg_rngs_tp = {}
        super().add_parameter(
            name="initial_valid_ranges",
            label="initial valid ranges",
            docstring="",
            set_cmd=self.set_initial_valid_ranges,
            get_cmd=self.get_initial_valid_ranges,
            initial_value=init_valid_ranges_renamed,
            vals=vals.Dict(),
        )



# TODO: Deal with ohmics?

        # if ohmic_parameters is not None:
        #     for ohmic_id, ohm_param in ohmic_parameters.items():
        #         for required_param in required_parameter_fields:
        #             assert ohm_param.get(required_param) is not None
        #         alias = f"ohmic_{ohmic_id}"
        #         ohm_param["ohmic_id"] = ohmic_id
        #         ohmic = Ohmic(
        #             parent=self,
        #             name=alias,
        #             **ohm_param,
        #         )
        #         super().add_submodule(alias, ohmic)
        #         self.ohmics.append(ohmic)


        super().add_parameter(
            name="quality",
            label="device quality",
            docstring="",
            set_cmd=None,
            get_cmd=None,
            initial_value=0,
            vals=vals.Numbers(),
        )
        if normalization_constants is not None:
            self._normalization_constants = normalization_constants
        else:
            self._normalization_constants = {
                key: (0, 1) for key in ["dc_current", "rf", "dc_sensor"]
            }

        super().add_parameter(
            name="normalization_constants",
            label="normalization constants",
            docstring=(
                "Signal measured with all gates at  zero. Used as "
                "normalization during data post processing"
            ),
            set_cmd=self.set_normalization_constants,
            get_cmd=self.get_normalization_constants,
            initial_value=self._normalization_constants,
            vals=vals.Dict(),
        )

    def _initialize_channel_lists(self, channels_input_mapping):
        gate_dict = {}
        ohmic_dict = {}
        for channel_name in channels_input_mapping.keys():
            if channel_name != "type":
                channel = getattr(self, channel_name)
                if channel.gate_id() is not None:
                    gate_dict[channel.gate_id()] = channel
                elif channel.ohmic_id() is not None:
                    ohmic_dict[channel.ohmic_id()] = channel
        gates_list = []
        for gate_id in range(0, len(gate_dict)):
            gates_list.append(gate_dict[gate_id])
        ohmics_list = []
        for ohmic_id in range(0, len(ohmic_dict)):
            ohmics_list.append(ohmic_dict[ohmic_id])
        return gates_list, ohmics_list


    def get_normalization_constants(self) -> Dict[str, Tuple[float, float]]:
        """"""
        return self._normalization_constants

    def set_normalization_constants(
        self,
        new_normalization_constant: Dict[str, Tuple[float, float]],
    ) -> None:
        """"""
        assert isinstance(new_normalization_constant, dict)
        for key in new_normalization_constant.keys():
            assert isinstance(new_normalization_constant[key], tuple) or isinstance(
                new_normalization_constant[key], list
            )
        self._normalization_constants.update(new_normalization_constant)

    def get_initial_valid_ranges(self) -> vltg_rngs_tp:
        """"""
        return copy.deepcopy(self._initial_valid_ranges)

    def set_initial_valid_ranges(self, new_valid_ranges: vltg_rngs_tp) -> None:
        """
        will update existing dict, not simply over write
        and set current valid ranges to ranges stored in initial_valid_ranges
        """
        self._initial_valid_ranges.update(new_valid_ranges)

    def ground_gates(self) -> None:
        for gate in self.gates:
            gate.ground()
            logger.info("DeviceChannel {} grounded.".format(gate.name))

    def float_ohmics(self) -> None:
        for ohmic in self.ohmics:
            ohmic.float_relay()
            logger.info("Ohmic {} floating.".format(ohmic.name))


    def get_gate_status(
        self,
    ) -> Dict[str, Dict[str, Union[Tuple[float, float], float]]]:
        """"""
        current_gate_status: Dict[
            str, Dict[str, Union[Tuple[float, float], float]]
        ] = {}
        for gate in self.gates:
            current_gate_status[gate.name] = {}
            rng = gate.current_valid_range()
            current_gate_status[gate.name]["current_valid_range"] = rng
            current_gate_status[gate.name]["voltage"] = gate.voltage()

        return current_gate_status

    def all_gates_to_highest(self) -> None:
        """
        Set all gate voltages to most positive voltage allowed.
        Will use ramp if gate.use_ramp is set to True
        """
        for gate in self.gates:
            gate.voltage(gate.safety_range()[1])

    def all_gates_to_lowest(self) -> None:
        """
        Set all gate voltages to most negative voltage allowed.
        Will use ramp if gate.use_ramp is set to True
        """
        for gate in self.gates:
            gate.voltage(gate.safety_range()[0])
