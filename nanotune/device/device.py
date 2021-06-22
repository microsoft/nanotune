import copy
import logging
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping, Sequence,
)
from enum import Enum
from collections import namedtuple
from functools import partial
import numpy as np
from dataclasses import dataclass
import qcodes as qc
from qcodes import validators as vals
from qcodes.station import Station

from nanotune.device.device_channel import DeviceChannel
if not qc.__version__.startswith('0.27'):
    from nanotune.device.delegate_channel_instrument import DelegateChannelInstrument as DelegateInstrument
else:
    from qcodes.instrument.delegate import DelegateInstrument  # type: ignore


logger = logging.getLogger(__name__)
READOUTMETHODS = ['transport', 'sensing', 'rf']

nrm_cnst_tp = Mapping[str, Tuple[float, float]]
voltage_range_type = Dict[int, Sequence[float]]
# TODO: Use dataclass for normalization constants
# @dataclass
# class NormalizationConstants:
#     transport: Sequence[float]
#     sensing: Sequence[float]
#     rf: Sequence[float]


def readout_formatter(
    *values: Any,
    param_names: List[str],
    name: str,
    **kwargs: Any,
) -> Any:
    return namedtuple(name, param_names)(*values, **kwargs)


class Device(DelegateInstrument):
    """
    device_type: str, e.g 'doubledot'
    readout = {
        'transport': qc.Parameter,
        'sensing': qc.Parameter,
        'rf': qc.Parameter,
    }

    """

    def __init__(
        self,
        name: str,
        station: Station,
        parameters: Optional[
            Union[Mapping[str, Sequence[str]], Mapping[str, str]]] = None,
        channels: Optional[
            Union[Mapping[str, Mapping[str, Any]], Mapping[str, str]]] = None,
        readout: Optional[Mapping[str, str]] = None,
        initial_values: Optional[Mapping[str, Any]] = None,
        set_initial_values_on_load: bool = False,
        device_type: Optional[str] = '',
        initial_valid_ranges: Optional[Mapping[str, Sequence[str]]] = None,
        current_valid_ranges: Optional[Mapping[str, Sequence[str]]] = None,
        normalization_constants: Optional[nrm_cnst_tp] = None,
        transition_voltages: Optional[Mapping[str, float]] = None,
        **kwargs,
    ) -> None:

        channels = _add_station_and_label_to_channel_init(station, channels)

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
        if channels is not None:
            (self.gates,
             self.ohmics,
             self._gate_labels) = self.initialize_channel_lists(channels)
        else:
            self.gates, self.ohmics = [], []

        if readout is None:
            self.readout = None
        else:
            param_names, paths = list(zip(*list(readout.items())))
            for param_name in param_names:
                if param_name not in READOUTMETHODS:
                    raise KeyError(f"Invalid readout method key. Use one of \
                        {READOUTMETHODS}")
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
                gate_id = gate.gate_id
                init_valid_ranges_renamed[gate_id] = gate.safety_voltage_range()
        else:
            init_valid_ranges_renamed = self.rename_gate_identifier(
                initial_valid_ranges
            )
            init_valid_ranges_renamed = self._fill_missing_voltage_ranges(
                init_valid_ranges_renamed)

        self._initial_valid_ranges: voltage_range_type = init_valid_ranges_renamed
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
                "Signal measured with all gates at  zero. Used as \
                normalization during data post processing"
            ),
            set_cmd=self.set_normalization_constants,
            get_cmd=self.get_normalization_constants,
            initial_value=self._normalization_constants,
            vals=vals.Dict(),
        )

        if current_valid_ranges is None:
            current_valid_ranges_renamed = init_valid_ranges_renamed
        else:
            current_valid_ranges_renamed = self.rename_gate_identifier(
                current_valid_ranges
            )
            current_valid_ranges_renamed = self._fill_missing_voltage_ranges(
                current_valid_ranges_renamed)

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
                [gate.gate_id for gate in self.gates], np.nan
            )
        else:
            transition_voltages_renamed = self.rename_gate_identifier(
                transition_voltages
            )
            defaults = {gate.gate_id: None for gate in self.gates}
            defaults.update(transition_voltages_renamed)
            transition_voltages_renamed = defaults
            # transition_voltages_renamed = self._fill_missing_voltage_ranges(
            #     current_valid_ranges_renamed, None)

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
            if not isinstance(new_normalization_constant[key], Sequence):
                raise TypeError('Wrong normalization constant item type, \
                     expect list or tuple.')
            if key not in READOUTMETHODS:
                raise KeyError(f'Invalid key, use one of {READOUTMETHODS}')
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
        current_valid_ranges = self.current_valid_ranges()
        for gate in self.gates:
            current_gate_status[gate.label] = {}
            rng = current_valid_ranges[gate.gate_id]
            current_gate_status[gate.label]["current_valid_range"] = rng
            current_gate_status[gate.label]["voltage"] = gate.voltage()

        return current_gate_status

    def all_gates_to_highest(self) -> None:
        """
        Set all gate voltages to most positive voltage allowed.
        Will use ramp if gate.use_ramp is set to True
        """
        for gate in self.gates:
            gate.voltage(gate.safety_voltage_range()[1])

    def all_gates_to_lowest(self) -> None:
        """
        Set all gate voltages to most negative voltage allowed.
        Will use ramp if gate.use_ramp is set to True
        """
        for gate in self.gates:
            gate.voltage(gate.safety_voltage_range()[0])

    def get_initial_valid_ranges(self) -> voltage_range_type:
        """"""
        return copy.deepcopy(self._initial_valid_ranges)

    def set_initial_valid_ranges(self, new_range) -> None:
        """ """
        self._initial_valid_ranges = self.voltage_range_setter(
            self._initial_valid_ranges, new_range
        )

    def get_current_valid_ranges(self) -> voltage_range_type:
        """"""
        return copy.deepcopy(self._current_valid_ranges)

    def set_current_valid_ranges(self, new_range) -> None:
        """ """
        self._current_valid_ranges = self.voltage_range_setter(
            self._current_valid_ranges, new_range
        )

    def voltage_range_setter(
        self,
        voltage_ranges: voltage_range_type,
        new_sub_dict: voltage_range_type,
    ) -> voltage_range_type:
        new_voltage_ranges = copy.deepcopy(voltage_ranges)
        for gate_identifier, new_range in new_sub_dict.items():
            gate_id = self.get_gate_id(gate_identifier)
            if gate_id is None:
                raise ValueError(f'Gate {gate_identifier} has not gate_id.')
            sfty_range = self.gates[gate_id].safety_voltage_range()
            new_range = self.check_and_update_new_voltage_range(
                new_range, sfty_range
            )
            new_voltage_ranges.update({gate_id: new_range})

        return new_voltage_ranges

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
            gate_id = self.get_gate_id(gate_identifier)
            sfty_range = self.gates[gate_id].safety_voltage_range()
            if new_T is not None:
                if new_T > sfty_range[1]:
                    new_T = sfty_range[1]
                    logger.warning(
                        f"Setting invalid transition voltage for \
                        {self._gate_labels[gate_id]}.\
                        Taking upper safety voltage. "
                    )
                if new_T < sfty_range[0]:
                    new_T = sfty_range[0]
                    logger.warning(
                        f"Setting invalid transition voltage for \
                        {self._gate_labels[gate_id]}.\
                        Taking lower safety voltage. "
                    )
            self._transition_voltages.update({gate_id: new_T})

    def rename_gate_identifier(
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
            gate_id = self.get_gate_id(gate_ref)
            if gate_id is None:
                raise ValueError(f'Gate {gate_ref} has not gate_id.')
            new_dict[gate_id] = param
        return new_dict

    def initialize_channel_lists(self, channels_input_mapping):
        gate_dict = {}
        ohmic_dict = {}
        gate_labels = {}
        _ = channels_input_mapping.pop("type", None)
        for channel_name in channels_input_mapping.keys():
            channel = getattr(self, channel_name)
            if channel.gate_id is not None:
                gate_dict[channel.gate_id] = channel
                gate_labels[channel.gate_id] = channel_name
            elif channel.ohmic_id is not None:
                ohmic_dict[channel.ohmic_id] = channel
        gates_list = []
        for gate_id in range(0, len(gate_dict)):
            gates_list.append(gate_dict[gate_id])
        ohmics_list = []
        for ohmic_id in range(0, len(ohmic_dict)):
            ohmics_list.append(ohmic_dict[ohmic_id])
        return gates_list, ohmics_list, gate_labels

    def check_and_update_new_voltage_range(
        self,
        new_range: Sequence[float],
        safety_range: Sequence[float],
    ) ->Sequence[float]:
        """ """
        if not isinstance(new_range, Sequence) or not len(new_range) == 2:
            raise ValueError('Wrong voltage range type.')
        new_range = sorted(new_range)
        if new_range[1] > safety_range[1]:
            new_range[1] = safety_range[1]
            logger.warning(
                "New range out of safety range. Taking upper safety limit."
            )
            if new_range[0] > safety_range[1]:
                raise ValueError(
                    "New lower voltage range is higher than upper safety \
                        limit. Something seems quite wrong."
                )
        if new_range[0] < safety_range[0]:
            new_range[0] = safety_range[0]
            logger.warning(
                "New range out of safety range. Taking lower safety limit."
            )
            if new_range[1] < safety_range[0]:
                raise ValueError("New upper voltage range is lower than upper \
                    safety limit. Something seems quite wrong."
                )
        return new_range

    def get_gate_id(
        self,
        gate_identifier: Union[Optional[int], str, DeviceChannel]
    ) -> Optional[int]:
        if isinstance(gate_identifier, DeviceChannel):
            if gate_identifier not in self.gates:
                raise ValueError("Gate not found in device.gates.")
            gate_id = gate_identifier.gate_id
        elif isinstance(gate_identifier, int):
            if gate_identifier not in self._gate_labels.keys():
                raise ValueError("Unknown gate ID - gate not found in \
                    device.gates.")
            gate_id = gate_identifier
        elif isinstance(gate_identifier, str):
            if gate_identifier not in self._gate_labels.values():
                raise ValueError("Unknown gate label - gate not found in \
                    device.gates.")
            gate_id = getattr(self, gate_identifier).gate_id
        else:
            raise ValueError("Invalid gate specifier. Use gate_id, label or \
                the channel itself.")
        return gate_id

    def _fill_missing_voltage_ranges(
        self,
        current_values: Dict[int, Sequence[float]],
    ) -> Dict[int, Sequence[float]]:
        for gate in self.gates:
            if gate.gate_id not in current_values.keys():
                current_values[gate.gate_id] = gate.safety_voltage_range()
        return current_values


def _add_station_and_label_to_channel_init(
    station: qc.Station,
    channels: Optional[Mapping[str, Union[str, Mapping[str, Any]]]] = None,
) -> Optional[Mapping[str, Union[str, Any]]]:
    if channels is None:
        return None
    for name, channel_value in channels.items():
        if isinstance(channel_value, Mapping):
            if 'station' not in channel_value.keys():
                channel_value['station'] = station  # type: ignore
            if 'label' not in channel_value.keys():
                channel_value['label'] = name  # type: ignore

    return channels