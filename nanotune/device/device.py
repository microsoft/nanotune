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
# from qcodes.instrument.delegate import DelegateInstrument
from qcodes.instrument.delegate.grouped_parameter import (
    DelegateGroup,
    DelegateGroupParameter,
    GroupedParameter,
)
from qcodes.station import Station

import nanotune as nt
from nanotune.device.device_channel import DeviceChannel
from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.device.delegate_channel_instrument import DelegateChannelInstrument


logger = logging.getLogger(__name__)

nrm_cnst_tp = Mapping[str, Tuple[float, float]]
vltg_rngs_tp = Dict[int, Tuple[Union[None, float], Union[None, float]]]

def readout_formatter(
    *values: Any,
    param_names: List[str],
    name: str,
    **kwargs: Any,
) -> Any:
    return namedtuple(name, param_names)(*values, **kwargs)


class Device(DelegateChannelInstrument):
    """
    device_type: str, e.g 'doubledot'
    readout_methods = {
        'transport': qc.Parameter,
        'sensing': qc.Parameter,
        'rf': qc.Parameter,
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
        current_valid_ranges: Optional[Mapping[str, Sequence[str]]] = None,
        normalization_constants: Optional[nrm_cnst_tp] = None,
        transition_voltages: Optional[Mapping[str, float]] = None,
        **kwargs,
    ) -> None:
        print(current_valid_ranges)

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

        if readout_parameters is None:
            param_names: Sequence[str] = []
            paths: Sequence[str] = []
        else:
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


        if initial_valid_ranges is None:
            init_valid_ranges_renamed: Dict[int, Any] = {}
            for gate in self.gates:
                gate_id = gate.gate_id()
                init_valid_ranges_renamed[gate_id] = gate.safety_range()
        else:
            init_valid_ranges_renamed = self._renamed_gate_key(
                initial_valid_ranges
            )

        self._initial_valid_ranges: vltg_rngs_tp = {}
        self.add_parameter(
            name="initial_valid_ranges",
            label="initial valid ranges",
            docstring="",
            set_cmd=self.set_initial_valid_ranges,
            get_cmd=self.get_initial_valid_ranges,
            initial_value=init_valid_ranges_renamed,
            vals=vals.Dict(),
        )

        self.add_parameter(
            name="quality",
            label="device quality",
            docstring="",
            set_cmd=None,
            get_cmd=None,
            initial_value=0,
            vals=vals.Numbers(),
        )
        if normalization_constants is not None:
            self._normalization_constants = dict(normalization_constants)
        else:
            self._normalization_constants = {
                key: (0, 1) for key in ["transport", "sensing"]
            }

        self.add_parameter(
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

        if current_valid_ranges is None:
            current_valid_ranges_renamed = init_valid_ranges_renamed
        else:
            current_valid_ranges_renamed = self._renamed_gate_key(
                current_valid_ranges
            )
        print(current_valid_ranges_renamed)
        self._current_valid_ranges = current_valid_ranges_renamed
        self.add_parameter(
            name="current_valid_ranges",
            label="current valid ranges",
            docstring="",
            set_cmd=self.set_current_valid_ranges,
            get_cmd=self.get_current_valid_ranges,
            initial_value=current_valid_ranges_renamed,
            vals=vals.Dict(),
        )
        if transition_voltages is None:
            transition_voltages_renamed = dict.fromkeys(
                [gate.gate_id() for gate in self.gates], np.nan
            )
        else:
            transition_voltages_renamed = self._renamed_gate_key(
                transition_voltages
            )

        self._transition_voltages = transition_voltages_renamed
        self.add_parameter(
            name="transition_voltages",
            label="gate transition voltages",
            docstring="",
            set_cmd=self.set_transition_voltages,
            get_cmd=self.get_transition_voltages,
            initial_value=transition_voltages_renamed,
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
        if not isinstance(new_normalization_constant, Mapping):
            raise TypeError('Wrong normalization constant type, expect dict.')
        for key in new_normalization_constant.keys():
            if (not isinstance(new_normalization_constant[key], Sequence)):
                raise TypeError('Wrong normalization constant item type, \
                     expect list or tuple.')
        self._normalization_constants.update(new_normalization_constant)

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

    def get_initial_valid_ranges(self) -> vltg_rngs_tp:
        """"""
        return copy.deepcopy(self._initial_valid_ranges)

    def set_initial_valid_ranges(self, new_valid_ranges: vltg_rngs_tp) -> None:
        """
        will update existing dict, not simply over write
        and set current valid ranges to ranges stored in initial_valid_ranges
        """
        self._initial_valid_ranges.update(new_valid_ranges)

    def get_current_valid_ranges(self) -> vltg_rngs_tp:
        """"""
        return copy.deepcopy(self._current_valid_ranges)

    def set_current_valid_ranges(
        self,
        new_current_ranges: Mapping[
            Union[DeviceChannel, int, str], Tuple[float, float]
        ],
    ) -> None:
        """
        will update existing dict, not simply over write
        and set current valid ranges to ranges stored in current_valid_ranges
        """

        msg = "Setting invalid current valid range for "
        msg2 = "Something is seriously wrong. New current valid range far off \
            safety range for "
        for gate_identifier, new_range in new_current_ranges.items():
            if not isinstance(new_range, list) or not len(new_range) == 2:
                raise TypeError('Wrong voltage range type.')
            new_range = sorted(new_range)
            gate = self._get_gate_from_identifier(gate_identifier)
            sfty_range = gate.safety_voltage_range()
            if new_range[1] > sfty_range[1]:
                new_range[1] = sfty_range[1]
                logger.warning(
                    msg + f"{gate.name}. Taking upper safety voltage."
                )
                if new_range[0] > sfty_range[1]:
                    raise ValueError(msg2 + gate.name)
            if new_range[0] < sfty_range[0]:
                new_range[0] = sfty_range[0]
                logger.warning(
                    msg+ f"{gate.name}. Taking lower safety voltage."
                )
                if new_range[1] < sfty_range[0]:
                    raise ValueError(msg2 + gate.name)

            self._current_valid_ranges.update({gate.gate_id(): new_range})

    def get_transition_voltages(self) -> Dict[int, float]:
        """"""
        return copy.deepcopy(self._transition_voltages)

    def set_transition_voltages(
        self,
        new_transition_voltages: Mapping[Union[DeviceChannel, int, str], float],
    ) -> None:
        """
        will update existing dict, not simply over write
        and set current valid ranges to ranges stored in current_valid_ranges
        """

        for gate_identifier, new_T in new_transition_voltages.items():
            gate = self._get_gate_from_identifier(gate_identifier)
            sfty_range = gate.safety_voltage_range()
            if new_T > sfty_range[1]:
                new_T = sfty_range[1]
                logger.warning(
                    f"Setting invalid transition voltage for {gate.name}.\
                    Taking upper safety voltage. "
                )
            if new_T < sfty_range[0]:
                new_T = sfty_range[0]
                logger.warning(
                    f"Setting invalid transition voltage for {gate.name}.\
                    Taking lower safety voltage. "
                )
            self._transition_voltages.update({gate.gate_id(): new_T})

    def _get_gate_from_identifier(
        self,
        reference: Union[DeviceChannel, int, str],
    ) -> DeviceChannel:
        """ """
        if isinstance(reference, str):
            gate = getattr(self, reference)
        elif isinstance(reference, int):
            gate = self.gates[reference]
        elif isinstance(reference, DeviceChannel):
            gate = reference
        else:
            raise ValueError('Unknown gate identifier')
        return gate

    def _renamed_gate_key(
        self,
        mapping_to_rename: Union[
            Mapping[DeviceChannel, Any],
            Mapping[str, Any],
            Mapping[int, Any]
        ],
    ) -> Dict[int, Any]:
        """ """
        new_dict = {}
        for gate_ref, param in mapping_to_rename.items():
            gate = self._get_gate_from_identifier(gate_ref)
            new_dict[gate.gate_id()] = param
        return new_dict


