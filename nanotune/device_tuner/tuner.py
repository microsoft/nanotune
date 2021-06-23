import copy
from dataclasses import dataclass, asdict
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import qcodes as qc
from qcodes import validators as vals
from qcodes.dataset.experiment_container import (load_last_experiment,
                                                 new_experiment)

import nanotune as nt
from nanotune.classification.classifier import Classifier
from nanotune.device.device import Device, voltage_range_type
from nanotune.device.device_channel import DeviceChannel
from nanotune.device_tuner.tuningresult import MeasurementHistory
from nanotune.tuningstages.gatecharacterization1d import GateCharacterization1D

logger = logging.getLogger(__name__)


@dataclass
class DataSettings:
    db_name: str = nt.config['db_name']
    db_folder: str = nt.config['db_folder']
    normalizations_constants: Optional[Dict[str, Tuple[float, float]]] = None
    experiment_id: Optional[int] = None
    segment_db_name: Optional[str] = None
    segment_db_folder: Optional[str] = None
    segment_experiment_id: Optional[int] = None


@dataclass
class SetpointSettings:
    voltage_precision: float
    parameters_to_sweep: Optional[List[qc.Parameter]] = None
    setpoint_method: Optional[
        Callable[[Any], Sequence[Sequence[float]]]] = None


@dataclass
class Classifiers:
    pinchoff: Optional[Classifier] = None
    singledot: Optional[Classifier] = None
    doubledot: Optional[Classifier] = None
    dotregime: Optional[Classifier] = None



@contextmanager
def set_back_voltages(gates: List[DeviceChannel]) -> Generator[None, None, None]:
    """Sets gates back to their respective voltage they are at before
    the contextmanager was called. If gates need to be set in a specific
    order, then this order needs to be respected on the list 'gates'.
    """
    initial_voltages = []
    for gate in gates:
        initial_voltages.append(gate.voltage())
    try:
        yield
    finally:
        for ig, gate in enumerate(gates):
            gate.voltage(initial_voltages[ig])


# @contextmanager
# def set_back_valid_ranges(
#     gates: List[DeviceChannel],
#     valid_ranges: voltage_range_type,
# ) -> Generator[None, None, None]:
#     """Sets gates back to their respective voltage they are at before
#     the contextmanager was called. If gates need to be set in a specific
#     order, then this order needs to be respected on the list 'gates'.
#     """
#     # valid_ranges = []
#     for gate in gates:
#         valid_ranges.append(gate.current_valid_range())
#     try:
#         yield
#     finally:
#         for ig, gate in enumerate(gates):
#             gate.current_valid_range(valid_ranges[ig])


class Tuner(qc.Instrument):
    """
    """

    def __init__(
        self,
        name: str,
        data_settings: DataSettings,
        classifiers: Classifiers,
        setpoint_settings: SetpointSettings,
    ) -> None:
        super().__init__(name)

        self.classifiers = classifiers

        self.data_settings = data_settings
        self.setpoint_settings = setpoint_settings

    @property
    def setpoint_settings(self) -> SetpointSettings:
        return self._setpoint_settings

    @property.setter
    def setpoint_settings(self, new_settings: SetpointSettings) -> None:
        self._setpoint_settings = new_settings
        self.metadata.update({'setpoint_settings': asdict(new_settings)})

    @property
    def data_settings(self) -> DataSettings:
        return self._setpoint_settings

    @property.setter
    def data_settings(self, new_settings: DataSettings) -> None:
        self._data_settings = new_settings
        self.metadata.update({'data_settings': asdict(new_settings)})

    def update_normalization_constants(self, device: Device):
        """
        Get the maximum and minimum signal measured of a device, for all
        readout methods specified in device.readout_methods. It will first
        set all gate voltages to their respective most negative allowed values
        and then to their most positive allowed ones.

        Puts device back into state where it was before, gates are set in the
        order as they are set in device.gates.
        """
        available_readout_methods = {
            r_type: qc_param
            for (r_type, qc_param) in device.readout_methods().items()
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
        device: Device,
        gates: List[DeviceChannel],
        use_safety_ranges: bool = False,
        comment: Optional[str] = None,
    ) -> MeasurementHistory:
        """
        Characterize multiple gates.
        It does not set any voltages.

        returns instance of MeasurementHistory
        """

        if comment is None:
            comment = f"Characterizing {gates}."
        if "pinchoff" not in self.classifiers.keys():
            raise KeyError("No pinchoff classifier found.")

        with set_back_valid_ranges(gates):
            if use_safety_ranges:
                for gate in gates:
                    gate.current_valid_range(gate.safety_range())

            measurement_result = MeasurementHistory(device.name)
            with self.device_specific_settings(device):
                for gate in gates:
                    setpoint_settings = copy.deepcopy(self.data_settings)
                    setpoint_settings.parameters_to_sweep = [gate.voltage]

                    stage = GateCharacterization1D(
                        data_settings=self.data_settings,
                        setpoint_settings=setpoint_settings,
                        readout_methods=device.readout_methods(),
                        classifier=self.classifiers.pinchoff,
                    )
                    tuningresult = stage.run_stage()
                    tuningresult.status = device.get_gate_status()
                    tuningresult.comment = comment
                    measurement_result.add_result(
                        tuningresult,
                        "characterization_" + gate.name,
                    )

        return measurement_result

    def measure_initial_ranges(
        self,
        device: Device,
        gate_to_set: DeviceChannel,
        gates_to_sweep: List[DeviceChannel],
        voltage_step: float = 0.2,
    ) -> Tuple[Tuple[float, float], MeasurementHistory]:
        """
        Estimate the default voltage range to consider

        Args:
            gate_to_set (nt.DeviceChannel):
            gates_to_sweep (list)
            voltage_step (flaot)
        Returns:
            tuple(float, float):
            MeasurementHistory:
        """
        if "pinchoff" not in self.classifiers.keys():
            raise KeyError("No pinchoff classifier found.")

        device.all_gates_to_highest()

        measurement_result = MeasurementHistory(device.name)
        layout_ids = [g.layout_id() for g in gates_to_sweep]
        skip_gates = dict.fromkeys(layout_ids, False)

        v_range = gate_to_set.safety_range()
        n_steps = int(abs(v_range[0] - v_range[1]) / voltage_step)

        with self.device_specific_settings(device):
            v_steps = np.linspace(np.max(v_range), np.min(v_range), n_steps)
            for voltage in v_steps:
                gate_to_set.voltage(voltage)

                for gate in gates_to_sweep:
                    if not skip_gates[gate.layout_id()]:
                        setpoint_sets = copy.deepcopy(self.data_settings)
                        setpoint_sets["parameters_to_sweep"] = [gate.voltage]
                        stage = GateCharacterization1D(
                            data_settings=self.data_settings,
                            setpoint_settings=setpoint_sets,
                            readout_methods=device.readout_methods(),
                            classifier=self.classifiers.pinchoff,
                        )
                        tuningresult = stage.run_stage()
                        tuningresult.status = device.get_gate_status()
                        measurement_result.add_result(
                            tuningresult,
                            f"characterization_{gate.name}",
                        )
                        if tuningresult.success:
                            skip_gates[gate.layout_id()] = True
                            last_gate = gate.layout_id()

                if all(skip_gates.values()):
                    break
        min_voltage = gate_to_set.voltage()

        # Swap top_barrier and last barrier to pinch off agains it to
        # determine opposite corner of valid voltage space.
        gate_to_set.parent.all_gates_to_highest()

        v_range = device.gates[last_gate].safety_range()
        n_steps = int(abs(v_range[0] - v_range[1]) / voltage_step)
        setpoint_settings = copy.deepcopy(self.data_settings)
        setpoint_settings.parameters_to_sweep = [gate_to_set.voltage]
        with self.device_specific_settings(device):
            v_steps = np.linspace(np.max(v_range), np.min(v_range), n_steps)
            for voltage in v_steps:

                device.gates[last_gate].voltage(voltage)
                stage = GateCharacterization1D(
                    data_settings=self.data_settings,
                    setpoint_settings=setpoint_settings,
                    readout_methods=device.readout_methods(),
                    classifier=self.classifiers.pinchoff,
                )

                tuningresult = stage.run_stage()
                tuningresult.status = device.get_gate_status()
                measurement_result.add_result(
                    tuningresult,
                    f"characterization_{gate.name}",
                )
                if tuningresult.success:
                    L = tuningresult.features["low_voltage"]
                    L = round(L - 0.1 * abs(L), 2)
                    min_rng = self.top_barrier.safety_range()[0]
                    max_voltage = np.max([min_rng, L])
                    break

        gate_to_set.parent.all_gates_to_highest()

        return (min_voltage, max_voltage), measurement_result

    @contextmanager
    def device_specific_settings(
        self,
        device: Device,
    ) -> Generator[None, None, None]:
        """Add device relevant readout settings

        Returns:
            Generator yielding nothing.
        """

        copy.deepcopy(self.data_settings)
        self.data_settings(
            {"normalization_constants": device.normalization_constants()},
        )
        try:
            yield
        finally:
            del self._data_settings.normalization_constants

