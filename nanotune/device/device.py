import os
import logging
import copy

import numpy as np
from typing import List, Optional, Dict, Tuple, Sequence, Any, Union
from operator import itemgetter

import qcodes as qc
from qcodes import validators as vals

import nanotune as nt
from nanotune.device.gate import Gate
from nanotune.device.ohmic import Ohmic
from nanotune.device_tuner.tuningresult import TuningResult

logger = logging.getLogger(__name__)

nrm_cnst_tp = Dict[str, Tuple[float, float]]
vltg_rngs_tp = Dict[int, Tuple[Union[None, float], Union[None, float]]]


class Device(qc.Instrument):
    """
    The Device class represents a physical sample with gates, ohmics and readout

    Attributes:

        name (str): string identifier. Should be valid, i.e. no spaces or
            special characters.
        device_type (str): One of the supported device types, e.g.
            'doubledot_2D', which are defined in config.json.
        gates (list): List of nt.Gate instances.
        sensor_gates (list): List of nt.Gate instances.
        ohmics (list): List of nt.Ohmics instances.
        initial_valid_ranges (list): List of valid voltages ranges if known in
            advance.
        quality (bool): Whether or not a device is working and should be tuned.
            To be determined trough a characterization and saved to metadata.
        normalization_constants (dict):
            {'dc_current': <float>,
                'dc_sensor': <float>,
                'rf': <float>}
        readout_methods (dict):
                {'dc_current': qc.Parameter,
                'dc_sensor': qc.Parameter,
                'rf': qc.Parameter}
        measurement_options (dict):
                {'dc_current': {
                    'delay': <float>,
                    'inter_delay': <float>}
                'dc_sensor': {
                    'delay': <float>,
                    'inter_delay': <float>}
                'rf': {
                    'delay': <float>,
                    'inter_delay': <float>}}

    Methods:
        getters and setters for qcodes parameters
        all_gates_to_highest:
        all_gates_to_zero:
    """
    def __init__(
        self,
        name: str,
        device_type: str,
        readout_methods: Optional[Dict[str, qc.Parameter]] = None,
        gate_parameters: Optional[Dict[int, Any]] = None,
        ohmic_parameters: Optional[Dict[int, Any]] = None,
        sensor_parameters: Optional[Dict[int, Any]] = None,
        measurement_options: Optional[Dict[str, Dict[str, Any]]] = None,
        initial_valid_ranges: Optional[vltg_rngs_tp] = None,
        normalization_constants: Optional[nrm_cnst_tp] = None,
    ) -> None:
        """
        Args:
            gate_parameters (dict):
                    {layout_id: int = {
                        'channel_id': int,
                        'dac_instrument': DACInterface,
                        'label': str,
                        'safety_range': Tuple[float, float]}}
            ohmic_parameters (dict):
                    {ohmic_id: int ={
                        'dac_instrument': DACInterface,
                        'channel_id': int,
                        'ohmic_id': int,
                        'label': str}}
            sensor_parameters (dict):
                    {layout_id: int = {
                        'channel_id': int,
                        'dac_instrument': DACInterface,
                        'label': str,
                        'safety_range': Tuple[float, float]}}
        """
        super().__init__(name)

        super().add_parameter(
            name="device_type",
            label="device type",
            docstring="",
            set_cmd=None,
            get_cmd=None,
            initial_value=device_type,
            vals=vals.Strings(),
        )

        all_meths = nt.config['core']['readout_methods']
        if readout_methods is not None:
            assert list(set(all_meths).intersection(readout_methods.keys()))
        else:
            readout_methods = {}
        self._readout_methods = readout_methods
        super().add_parameter(
            name="readout_methods",
            label=f"{name} readout methods",
            docstring="readout methods to use for measurements",
            set_cmd=self.set_readout_methods,
            get_cmd=self.get_readout_methods,
            initial_value=readout_methods,
            vals=vals.Dict(),
        )

        self._measurement_options = measurement_options
        super().add_parameter(
            name="measurement_options",
            label=f"{name} measurement options",
            docstring="readout methods to use for measurements",
            set_cmd=self.set_measurement_options,
            get_cmd=self.get_measurement_options,
            initial_value=measurement_options,
            vals=vals.Dict(),
        )

        self.layout = nt.config["device"][self.device_type()]
        self.gates: List[Gate] = []
        self.sensor_gates: List[Gate] = []
        self.ohmics: List[Ohmic] = []
        required_parameter_fields = ['channel_id', 'dac_instrument']
        if gate_parameters is not None:
            for layout_id, gate_param in gate_parameters.items():
                for required_param in required_parameter_fields:
                    assert gate_param.get(required_param) is not None
                alias = self.layout[layout_id]
                gate_param['layout_id'] = layout_id
                gate_param['use_ramp'] = True
                gate = Gate(
                    parent=self,
                    name=alias,
                    **gate_param,
                )
                super().add_submodule(alias, gate)
                self.gates.append(gate)

        if sensor_parameters is not None:
            for layout_id, sens_param in sensor_parameters.items():
                for required_param in required_parameter_fields:
                    assert sens_param.get(required_param) is not None
                alias = self.layout[layout_id]
                sens_param['layout_id'] = layout_id
                sens_param['use_ramp'] = True
                gate = Gate(
                    parent=self,
                    name=alias,
                    **sens_param,
                )
                super().add_submodule(alias, gate)
                self.sensor_gates.append(gate)

        if ohmic_parameters is not None:
            for ohmic_id, ohm_param in ohmic_parameters.items():
                for required_param in required_parameter_fields:
                    assert ohm_param.get(required_param) is not None
                alias = f"ohmic_{ohmic_id}"
                ohm_param['ohmic_id'] = ohmic_id
                ohmic = Ohmic(
                    parent=self,
                    name=alias,
                    **ohm_param,
                )
                super().add_submodule(alias, ohmic)
                self.ohmics.append(ohmic)

        if initial_valid_ranges is None:
            initial_valid_ranges = {}
            for gate in self.gates:
                initial_valid_ranges[gate.layout_id()] = gate.safety_range()

        self._initial_valid_ranges: vltg_rngs_tp = {}
        super().add_parameter(
            name="initial_valid_ranges",
            label="initial valid ranges",
            docstring="",
            set_cmd=self.set_initial_valid_ranges,
            get_cmd=self.get_initial_valid_ranges,
            initial_value=initial_valid_ranges,
            vals=vals.Dict(),
        )

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
            label="open circuit signal",
            docstring=(
                "Signal measured with all gates at  zero. Used as "
                "normalization during data post processing"
            ),
            set_cmd=self.set_normalization_constants,
            get_cmd=self.get_normalization_constants,
            initial_value=self._normalization_constants,
            vals=vals.Dict(),
        )

    def snapshot_base(
        self,
        update: Optional[bool] = True,
        params_to_skip_update: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        snap = super().snapshot_base(update, params_to_skip_update)

        name = "readout_methods"
        snap["parameters"][name] = {}
        for read_type, qc_param in self.readout_methods().items():
            if qc_param is not None:
                sub_snap = qc_param.snapshot()
                snap["parameters"][name][read_type] = sub_snap
            else:
                snap["parameters"][name][read_type] = {}

        return snap

    def get_normalization_constants(self) -> Dict[str, Tuple[float, float]]:
        """
        """
        return self._normalization_constants

    def set_normalization_constants(
        self,
        new_normalization_constant: Dict[str, Tuple[float, float]],
        ) -> None:
        """"""
        assert isinstance(new_normalization_constant, dict)
        for key in new_normalization_constant.keys():
            assert (isinstance(new_normalization_constant[key], tuple) or
                    isinstance(new_normalization_constant[key], list))
        self._normalization_constants.update(new_normalization_constant)

    def get_readout_methods(self) -> Dict[str, qc.Parameter]:
        """
        Update all attributed using normalization_constants?
        """
        return self._readout_methods

    def set_readout_methods(
        self,
        readout_methods: Dict[str, qc.Parameter],
    ) -> None:
        """"""
        self._readout_methods.update(readout_methods)

    def get_measurement_options(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Update all attributed using normalization_constants?
        """
        return self._measurement_options

    def set_measurement_options(
        self,
        measurement_options: Dict[str, Dict[str, Any]],
    ) -> None:
        """"""
        if self._measurement_options is None:
            self._measurement_options = {}
        self._measurement_options.update(measurement_options)

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
            logger.info("Gate {} grounded.".format(gate.name))

    def float_ohmics(self) -> None:
        for ohmic in self.ohmics:
            ohmic.float_relay()
            logger.info("Ohmic {} floating.".format(ohmic.name))

    def get_gate_status(
            self) -> Dict[str, Dict[str, Union[Tuple[float, float], float]]]:
        """"""
        current_gate_status: Dict[str,
            Dict[str, Union[Tuple[float, float], float]]] = {}
        for gate in self.gates:
            current_gate_status[gate.name] = {}
            rng = gate.current_valid_range()
            current_gate_status[gate.name]['current_valid_range'] = rng
            current_gate_status[gate.name]['dc_voltage'] = gate.dc_voltage()

        return current_gate_status

    def all_gates_to_highest(self) -> None:
        """
        Set all gate voltages to most positive voltage allowed.
        Will use ramp if gate.use_ramp is set to True
        """
        for gate in self.gates:
            gate.dc_voltage(gate.safety_range()[1])

    def all_gates_to_lowest(self) -> None:
        """
        Set all gate voltages to most negative voltage allowed.
        Will use ramp if gate.use_ramp is set to True
        """
        for gate in self.gates:
            gate.dc_voltage(gate.safety_range()[0])
