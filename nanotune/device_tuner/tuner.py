import os
import copy
import logging
import time
import datetime
from typing import (List, Optional, Dict, Tuple, Sequence, Callable, Any,
                    Union, Generator)
from functools import partial
from contextlib import contextmanager
import numpy as np

import qcodes as qc
from qcodes import validators as vals
from qcodes.dataset.experiment_container import (load_last_experiment,
                                                 load_experiment,
                                                 new_experiment)

import nanotune as nt
from nanotune.device.device import Device as Nt_Device
from nanotune.device_tuner.tuningresult import MeasurementHistory
from nanotune.classification.classifier import Classifier
from nanotune.tuningstages.gatecharacterization1d import GateCharacterization1D
from nanotune.device.gate import Gate
from nanotune.utils import flatten_list
logger = logging.getLogger(__name__)
DATA_DIMS = {
    'gatecharacterization1d': 1,
    'chargediagram': 2,
    'coulomboscillations': 1,
}


@contextmanager
def set_back_voltages(gates: List[Gate]) -> Generator[None, None, None]:
    """Sets gates back to their respective dc_voltage they are at before
    the contextmanager was called. If gates need to be set in a specific
    order, then this order needs to be respected on the list 'gates'.
    """
    initial_voltages = []
    for gate in gates:
        initial_voltages.append(gate.dc_voltage())
    try:
        yield
    finally:
        for ig, gate in enumerate(gates):
            gate.dc_voltage(initial_voltages[ig])


@contextmanager
def set_back_valid_ranges(gates: List[Gate]) -> Generator[None, None, None]:
    """Sets gates back to their respective dc_voltage they are at before
    the contextmanager was called. If gates need to be set in a specific
    order, then this order needs to be respected on the list 'gates'.
    """
    valid_ranges = []
    for gate in gates:
        valid_ranges.append(gate.current_valid_range())
    try:
        yield
    finally:
        for ig, gate in enumerate(gates):
            gate.current_valid_range(valid_ranges[ig])


class Tuner(qc.Instrument):
    """
    classifiers = {
        'pinchoff': Optional[Classifier],
        'singledot': Optional[Classifier],
        'doubledot': Optional[Classifier],
        'dotregime': Optional[Classifier],
    }
    data_settings = {
        'db_name': str,
        'db_folder': Optional[str],
        'qc_experiment_id': Optional[int],
        'segment_db_name': Optional[str],
        'segment_db_folder': Optional[str],
    }
    setpoint_settings = {
        'voltage_precision': float,
    }
    fit_options = {
        'pinchofffit': Dict[str, Any],
        'dotfit': Dict[str, Any],
    }

    """
    def __init__(
        self,
        name: str,
        data_settings: Dict[str, Any],
        classifiers: Dict[str, Classifier],
        setpoint_settings: Dict[str, Any],
        fit_options: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(name)

        self.classifiers = classifiers

        assert 'db_name' in data_settings.keys()
        if 'db_folder' in data_settings.keys():
            nt.set_database(
                data_settings['db_name'],
                db_folder=data_settings['db_folder'])
        else:
            nt.set_database(data_settings['db_name'])

        if data_settings.get('qc_experiment_id') is None:
            try:
                self.qcodes_experiment = load_last_experiment()
            except ValueError:
                logger.warning(
                    'No qcodes experiment found. Starting a new '
                    'one called "automated_tuning", with an unknown sample.'
                )
                self.qcodes_experiment = new_experiment(
                        "automated_tuning",
                        sample_name="unknown")
            exp_id = self.qcodes_experiment.exp_id
            data_settings['qc_experiment_id'] = exp_id

        self._data_settings = data_settings
        super().add_parameter(
            name="data_settings",
            label="data_settings",
            docstring="",
            set_cmd=self.update_data_settings,
            get_cmd=self.get_data_settings,
            initial_value=data_settings,
            vals=vals.Dict(),
        )
        if fit_options is None or not fit_options:
            fit_options = {
                key: {} for key in nt.config['core']['implemented_fits']
            }

        self._fit_options = fit_options
        super().add_parameter(
            name="fit_options",
            label="fit_options",
            docstring="",
            set_cmd=self.set_fit_options,
            get_cmd=self.get_fit_options,
            initial_value=fit_options,
            vals=vals.Dict(),
        )

        super().add_parameter(
            name="setpoint_settings",
            label="setpoint_settings",
            docstring="options for setpoint determination",
            set_cmd=None,
            get_cmd=None,
            initial_value=setpoint_settings,
            vals=vals.Dict(),
        )

    def update_normalization_constants(self, device: Nt_Device):
        """
        Get the maximum and minimum signal measured of a device, for all
        readout methods specified in device.readout_methods. It will first
        set all gate voltages to their respective most negative allowed values
        and then to their most positive allowed ones.

        Puts device back into state where it was before, gates are set in the
        order as they are set in device.gates.
        """
        available_readout_methods = {
            r_type: qc_param for (r_type, qc_param)
            in device.readout_methods().items()
            if qc_param is not None
            }

        normalization_constants = {
            key: [0.0, 1.0] for key in available_readout_methods.keys()
        }
        with set_back_voltages(device.gates):
            device.all_gates_to_lowest()
            for read_meth in available_readout_methods.keys():
                val = float(device.readout_methods()[read_meth].get())
                normalization_constants[read_meth][0] = val

            device.all_gates_to_highest()
            for read_meth in available_readout_methods.keys():
                val = float(device.readout_methods()[read_meth].get())
                normalization_constants[read_meth][1] = val
        device.normalization_constants(normalization_constants)

    def characterize_gates(
        self,
        gates: List[Gate],
        use_safety_ranges: bool = False,
        comment: Optional[str] = None,
    ) -> MeasurementHistory:
        """
        Characterize multiple gates.
        It does not set any voltages.

        returns instance of MeasurementHistory
        """

        device: Nt_Device = gates[0].parent

        if comment is None:
            comment = f'Characterizing {gates}.'
        if 'pinchoff' not in self.classifiers.keys():
            raise KeyError('No pinchoff classifier found.')

        with set_back_valid_ranges(gates):
            if use_safety_ranges:
                for gate in gates:
                    gate.current_valid_range(gate.safety_range())

            measurement_result = MeasurementHistory(device.name)
            with self.device_specific_settings(device):
                for gate in gates:
                    setpoint_settings = copy.deepcopy(self.setpoint_settings())
                    setpoint_settings['gates_to_sweep'] = [gate]

                    stage = GateCharacterization1D(
                        data_settings=self.data_settings(),
                        setpoint_settings=setpoint_settings,
                        readout_methods=device.readout_methods(),
                        classifier=self.classifiers['pinchoff'],
                        fit_options=self.fit_options()['pinchofffit'],
                        measurement_options=device.measurement_options(),
                    )
                    tuningresult = stage.run_stage()
                    tuningresult.gates_status = device.get_gate_status()
                    tuningresult.comment = comment
                    measurement_result.add_result(
                        tuningresult,
                        'characterization_' + gate.name,
                    )

        return measurement_result

    def measure_initial_ranges(
        self,
        gate_to_set: Gate,
        gates_to_sweep: List[Gate],
        voltage_step: float = 0.2,
    ) -> Tuple[Tuple[float, float], MeasurementHistory]:
        """
        Estimate the default voltage range to consider

        Args:
            gate_to_set (nt.Gate):
            gates_to_sweep (list)
            voltage_step (flaot)
        Returns:
            tuple(float, float):
            MeasurementHistory:
        """
        if 'pinchoff' not in self.classifiers.keys():
            raise KeyError('No pinchoff classifier found.')

        device: Nt_Device = gates_to_sweep[0].parent

        device.all_gates_to_highest()

        measurement_result = MeasurementHistory(device.name)
        layout_ids = [g.layout_id() for g in gates_to_sweep]
        skip_gates = dict.fromkeys(layout_ids, False)

        v_range = gate_to_set.safety_range()
        n_steps = int(abs(v_range[0] - v_range[1]) / voltage_step)

        with self.device_specific_settings(device):
            v_steps = np.linspace(np.max(v_range), np.min(v_range), n_steps)
            for voltage in v_steps:
                gate_to_set.dc_voltage(voltage)

                for gate in gates_to_sweep:
                    if not skip_gates[gate.layout_id()]:
                        setpoint_sets = copy.deepcopy(self.setpoint_settings())
                        setpoint_sets['gates_to_sweep'] = [gate]
                        stage = GateCharacterization1D(
                            data_settings=self.data_settings(),
                            setpoint_settings=setpoint_sets,
                            readout_methods=device.readout_methods(),
                            update_settings=False,
                            classifier=self.classifiers['pinchoff'],
                            fit_options=self.fit_options()['pinchofffit'],
                            measurement_options=device.measurement_options(),
                        )
                        tuningresult = stage.run_stage()
                        tuningresult.gates_status = device.get_gate_status()
                        measurement_result.add_result(
                            tuningresult,
                            f'characterization_{gate.name}',
                        )
                        if tuningresult.success:
                            skip_gates[gate.layout_id()] = True
                            last_gate = gate.layout_id()

                if all(skip_gates.values()):
                    break
        min_voltage = gate_to_set.dc_voltage()

        # Swap top_barrier and last barrier to pinch off agains it to
        # determine opposite corner of valid voltage space.
        gate_to_set.parent.all_gates_to_highest()

        v_range = device.gates[last_gate].safety_range()
        n_steps = int(abs(v_range[0] - v_range[1]) / voltage_step)
        setpoint_settings = copy.deepcopy(self.setpoint_settings())
        setpoint_settings['gates_to_sweep'] = [gate_to_set]
        with self.device_specific_settings(device):
            v_steps = np.linspace(np.max(v_range), np.min(v_range), n_steps)
            for voltage in v_steps:

                device.gates[last_gate].dc_voltage(voltage)
                stage = GateCharacterization1D(
                    data_settings=self.data_settings(),
                    setpoint_settings=setpoint_settings,
                    readout_methods=device.readout_methods(),
                    update_settings=False,
                    classifier=self.classifiers['pinchoff'],
                    fit_options=self.fit_options()['pinchofffit'],
                    measurement_options=device.measurement_options(),
                )

                tuningresult = stage.run_stage()
                tuningresult.gates_status = device.get_gate_status()
                measurement_result.add_result(
                    tuningresult,
                    f'characterization_{gate.name}',
                )
                if tuningresult.success:
                    L = tuningresult.features['low_voltage']
                    L = round(L - 0.1*abs(L), 2)
                    min_rng = self.top_barrier.safety_range()[0]
                    max_voltage = np.max([min_rng, L])
                    break

        gate_to_set.parent.all_gates_to_highest()

        return (min_voltage, max_voltage), measurement_result

    @contextmanager
    def device_specific_settings(
        self,
        device: Nt_Device,
    ) -> Generator[None, None, None]:
        """ Add device relevant readout settings

        Returns:
            Generator yielding nothing.
        """

        original_data_settings = copy.deepcopy(self.data_settings())
        self.data_settings(
            {'normalization_constants': device.normalization_constants()},
            )
        try:
            yield
        finally:
            del self._data_settings['normalization_constants']

    def set_fit_options(self, new_fit_options: Dict[str, Any]) -> None:
        """ """
        self._fit_options.update(new_fit_options)

    def get_fit_options(self) -> Dict[str, Any]:
        """ """
        return self._fit_options

    def get_data_settings(self) -> Dict[str, Any]:
        """ """
        return self._data_settings

    def update_data_settings(self, new_settings: Dict[str, Any]):
        self._data_settings.update(new_settings)
