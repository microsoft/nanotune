from __future__ import annotations
import copy
import logging
from enum import Enum
from typing import (
    Any, Dict, Optional, Sequence, Tuple, Union, Mapping, Sequence, List
)

import numpy as np
from dataclasses import asdict, dataclass
import qcodes as qc
from qcodes import validators as vals
from qcodes.station import Station

from nanotune.device.device_channel import DeviceChannel
if not qc.__version__.startswith('0.27'):
    from nanotune.device.delegate_channel_instrument import DelegateChannelInstrument as DelegateInstrument
else:
    from qcodes.instrument.delegate import DelegateInstrument  # type: ignore
from qcodes.instrument.delegate.grouped_parameter import GroupedParameter

logger = logging.getLogger(__name__)

nrm_cnst_tp = Mapping[str, Tuple[float, float]]
voltage_range_type = Dict[int, Sequence[float]]


@dataclass
class NormalizationConstants:
    """Container to hold normalization constant. They are the highest and
    lowest signals measured, corresponding to pinched-off and open signals.
    They are typically measured by setting all gates to their lowest allowed
    values to measure the lower bound, while the upper bound is measured with
    all gates set to their highest allowed voltages.

    Attributes:
        transport (Tuple[float, float]): constants for DC transport.
        sensing (Tuple[float, float]): constants for charge sensing (transport
            or other).
        rf (Tuple[float, float]): constants for reflectometry measurements.
    """
    transport: Tuple[float, float] = (0., 1.)
    sensing: Tuple[float, float] = (0., 1.)
    rf: Tuple[float, float] = (0., 1.)

    def update(
        self,
        new_constants: Union[
            Dict[str, Sequence[float]], NormalizationConstants],
    ) -> None:
        """Updates normalization constant. Raises an error if input is not a
        dict or NormalizationConstants instance.

        Args:
            new_constants (Dict or NormalizationConstants): new normalization
                constants. If they don't contain all constants, only those
                specified are updated with all other keeping their previous
                values.
        """
        if isinstance(new_constants, NormalizationConstants):
            new_constants_dict = asdict(new_constants)
        elif isinstance(new_constants, Dict):
            new_constants_dict = new_constants
        else:
            raise ValueError('Invalid normalization constants. Use \
                NormalizationConstants or a Dict instead.')

        for read_type, constant in new_constants_dict.items():
            if not isinstance(constant, Sequence):
                raise TypeError('Wrong normalization constant item type, \
                    expect list or tuple.')
            if not hasattr(self, read_type):
                raise KeyError(f'Invalid normalization constant identifier, \
                    use one of {self.__dataclass_fields__.keys()}')  # type: ignore
            setattr(self, read_type, tuple(constant))


@dataclass
class Readout:
    """Container grouping readout of a device.

    Attributes:
        transport (Tuple[float, float]): parameter to read out for DC transport.
        sensing (Tuple[float, float]): parameter to read out for charge sensing
            (transport or other).
        rf (Tuple[float, float]): parameter to read out for reflectometry
            measurements.
    """
    transport: Optional[Union[GroupedParameter, qc.Parameter]] = None
    sensing: Optional[Union[GroupedParameter, qc.Parameter]] = None
    rf: Optional[Union[GroupedParameter, qc.Parameter]] = None

    def available_readout(self) -> Dict[str, GroupedParameter]:
        """Gets readout parameters which are not None.

        Returns:
            Dict: mapping string identifier, e.g. "transport" onto a QCoDeS
                GroupedParameter.
        """
        param_dict = {}
        for field in Readout.__dataclass_fields__.keys():
            readout = getattr(self, field)
            if readout is not None:
                param_dict[field] = readout
        return param_dict

    def get_parameters(self) -> List[GroupedParameter]:
        """Gets list of parameters to read out.

        Returns:
            list: list of those GroupedParameters which are not None.
        """
        return list(self.available_readout().values())

    def as_name_dict(self) -> Dict[str, str]:
        """Gets readout parameter names.

        Returns:
            dict: mapping of string identifiers, e.g. 'transport' onto the
                GroupParameter's full name (also string).
        """
        param_dict = {}
        for field in Readout.__dataclass_fields__.keys():
            readout = getattr(self, field)
            if readout is not None:
                param_dict[field] = readout.full_name
        return param_dict


class ReadoutMethods(Enum):
    """Enumerates readout methods used in nanotune."""
    transport = 0
    sensing = 1
    rf = 2

    @classmethod
    def list(cls) -> List[int]:
        """Gets attribute values as a list."""
        return list(map(lambda c: c.value, cls))

    @classmethod
    def names(cls):
        """Gets list of attribute names (strings)."""
        return list(map(lambda c: c.name, cls))


class Device(DelegateInstrument):
    """Device abstraction with attributes and methods for (dot) tuning.

    Attributes:
        name (str): string identifier, used e.g. when saving tuning results.
        readout (Readout): readout parameters. One QCoDeS GroupedParameter for
            each readout method (transport, sensing, rf).
        gates (list[DeviceChannel]): list of gates; instances of DeviceChannel
        ohmics (list[DeviceChannel]): list of ohmics; instances of DeviceChannel
        <each gate>: each gate is added as an attribute with its name, eg.
            device.top_barrier.
        <each ohmic>: each ohmic is added as an attribute with its name, eg.
            device.left_ohmic.
        parameters (dict): mapping parameter names onto parameters, of all
            parameters.
        initial_valid_ranges (dict): dict mapping gate IDs into their initial
            valid ranges.
        transition_voltages (dict): dict mapping gate IDs into their transition
            voltages.
        current_valid_ranges (dict): dict mapping gate IDs into their current
            valid ranges.
        main_readout_method (ReadoutMethods): a ReadoutMethods item indicating
            which readout signal should be used for tuning decisions.
            Added to static metadata.
        normalization_constants (NormalizationConstants): normalization
            constants keeping track of highest and lowest signals recorded for
            all specified readout methods.
        quality (bool): quality of device, typically determined during
            characterization. Only good devices will be tuned.

    Main methods:
        ground_gates: grounds all gates.
        float_ohmics: sets relay of all ohmics to float.
        get_gate_status: returns dict mapping gate labels onto another dict
            with their current valid range and current voltage.
        all_gates_to_highest: sets all gates to their upper safety limit.
        all_gates_to_lowest: sets all gates to their lower safety limit.
        get_gate_id: returns ID of a gate based on either the gate or its label.
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
        main_readout_method: Union[ReadoutMethods, str] = ReadoutMethods.transport,
        initial_values: Optional[Mapping[str, Any]] = None,
        set_initial_values_on_load: bool = False,
        device_type: Optional[str] = '',
        initial_valid_ranges: Optional[Mapping[str, Sequence[str]]] = None,
        current_valid_ranges: Optional[Mapping[str, Sequence[str]]] = None,
        normalization_constants: Optional[
            Union[Dict[str, Sequence[float]], NormalizationConstants]] = None,
        transition_voltages: Optional[Mapping[str, float]] = None,
        **kwargs,
    ) -> None:
        """Device init method.

        Args:
            name (str): instrument/device name.
            station (qc.Station): station containing the real instrument that is used to get
                the endpoint parameters.
            parameters: mapping from the name of a parameter to the sequence
                of source parameters that it points to.
            channels:
            readout:
            main_readout_method (str or ReadoutMethods):
            set_initial_values_on_load: Flag to set initial values when the
                instrument is loaded. Defaults to False.
            device_type (str): type of device, e.g. 'doubledot_2D'.
            initial_valid_ranges (dict): if known, the valid ranges within
                which we need to tune.
            current_valid_ranges (dict): if known, the current ranges within
                which we need to tune. Can be used if the device needs to be
                (re-)loaded in the middle of a tune-up.
            normalization_constants (NormalizationConstants):
            transition_voltages
            initial_values: part kwargs. Default values to set on the delegate
                instrument's parameters. Defaults to None (no initial values are
                specified or
                set).
        """
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
             self._gate_labels,
             self._gates_dict,
             self._ohmics_dict) = self.initialize_channel_lists(channels)
        else:
            self.gates, self.ohmics = [], []
            self._gates_dict, self._ohmics_dict = {}, {}

        self.readout = Readout()
        if readout is not None:
            param_names, paths = list(zip(*list(readout.items())))
            for param_name, path in zip(param_names, paths):
                if param_name not in ReadoutMethods.names():
                    raise KeyError(f"Invalid readout method key. Use one of \
                        {ReadoutMethods.names()}")
                if not isinstance(path, list):
                    path = [path]
                super()._create_and_add_parameter(
                    param_name,
                    station,
                    path,
                )
                setattr(self.readout, param_name, getattr(self, param_name))
            self.metadata['readout'] = self.readout.as_name_dict()

        if isinstance(main_readout_method, str):
            self._main_readout_method = getattr(
                ReadoutMethods, main_readout_method)
        else:
            self._main_readout_method = main_readout_method
        self.metadata['main_readout_method'] = self._main_readout_method.name


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
            set_cmd=self._set_initial_valid_ranges,
            get_cmd=self._get_initial_valid_ranges,
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
        self._normalization_constants = NormalizationConstants()
        if normalization_constants is not None:
            self._normalization_constants.update(normalization_constants)

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
            set_cmd=self._set_current_valid_ranges,
            get_cmd=self._get_current_valid_ranges,
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


        self._transition_voltages = transition_voltages_renamed
        self.add_parameter(
            name="transition_voltages",
            label="gate transition voltages",
            docstring="",
            set_cmd=self._set_transition_voltages,
            get_cmd=self._get_transition_voltages,
            initial_value=transition_voltages_renamed,
            vals=vals.Dict(),
        )

    @property
    def normalization_constants(self) -> NormalizationConstants:
        return self._normalization_constants

    @normalization_constants.setter
    def normalization_constants(
        self,
        new_constants: Union[
            Dict[str, Sequence[float]], NormalizationConstants],
    ) -> None:
        self._normalization_constants.update(new_constants)
        self.metadata.update(
            {'normalization_constants': asdict(self._normalization_constants)}
        )

    @property
    def main_readout_method(self) -> ReadoutMethods:
        return self._main_readout_method

    @main_readout_method.setter
    def main_readout_method(self, readout_method):
        if not isinstance(readout_method, ReadoutMethods):
            raise ValueError("Unknown main readout method.")
        if getattr(self.readout, readout_method.name) is None:
            raise ValueError(
                f'Main readout method {readout_method} not found for ' \
                f'{self.name}'
            )
        self._main_readout_method

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

    def _get_initial_valid_ranges(self) -> voltage_range_type:
        """"""
        return copy.deepcopy(self._initial_valid_ranges)

    def _set_initial_valid_ranges(self, new_range) -> None:
        """ """
        self._initial_valid_ranges = self._voltage_range_setter(
            self._initial_valid_ranges, new_range, 'initial valid range',
        )

    def _get_current_valid_ranges(self) -> voltage_range_type:
        """"""
        return copy.deepcopy(self._current_valid_ranges)

    def _set_current_valid_ranges(self, new_range) -> None:
        """ """
        self._current_valid_ranges = self._voltage_range_setter(
            self._current_valid_ranges, new_range, 'current valid range',
        )

    def _voltage_range_setter(
        self,
        voltage_ranges: voltage_range_type,
        new_sub_dict: voltage_range_type,
        range_label: str = 'voltage range',
    ) -> voltage_range_type:
        new_voltage_ranges = copy.deepcopy(voltage_ranges)
        for gate_identifier, new_range in new_sub_dict.items():
            gate_id = self.get_gate_id(gate_identifier)
            if gate_id is None:
                raise ValueError(f'Gate {gate_identifier}: no gate_id.')
            sfty_range = self._gates_dict[gate_id].safety_voltage_range()
            new_range = self._check_and_update_new_voltage_range(
                new_range, sfty_range, self._gates_dict[gate_id].voltage(),
            )
            new_voltage_ranges.update({gate_id: new_range})
            logger.info(
                f"{self.gates[gate_id].name}: new {range_label} set \
                    to {new_range}"
            )

        return new_voltage_ranges

    def _get_transition_voltages(self) -> Dict[int, float]:
        """"""
        return copy.deepcopy(self._transition_voltages)

    def _set_transition_voltages(
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
        for gate_id in sorted(gate_dict.keys()):
            gates_list.append(gate_dict[gate_id])
        ohmics_list = []
        for ohmic_id in sorted(ohmic_dict.keys()):
            ohmics_list.append(ohmic_dict[ohmic_id])
        return gates_list, ohmics_list, gate_labels, gate_dict, ohmic_dict

    def _check_and_update_new_voltage_range(
        self,
        new_range: Sequence[float],
        safety_voltage_range: Sequence[float],
        current_voltage: float,
    ) ->Sequence[float]:
        """ """
        if not isinstance(new_range, Sequence) or not len(new_range) == 2:
            raise ValueError('Wrong voltage range type.')
        new_range = sorted(new_range)
        if current_voltage < new_range[0] or current_voltage > new_range[1]:
            logger.warning(
                "Current voltage not within new valid range."
            )
        if new_range[1] > safety_voltage_range[1]:
            new_range[1] = safety_voltage_range[1]
            logger.warning(
                "New range out of safety range. Taking upper safety limit."
            )
            if new_range[0] > safety_voltage_range[1]:
                raise ValueError(
                    "New lower voltage range is higher than upper safety \
                        limit. Something seems quite wrong."
                )
        if new_range[0] < safety_voltage_range[0]:
            new_range[0] = safety_voltage_range[0]
            logger.warning(
                "New range out of safety range. Taking lower safety limit."
            )
            if new_range[1] < safety_voltage_range[0]:
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
