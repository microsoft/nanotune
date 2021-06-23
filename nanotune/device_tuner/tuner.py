from dataclasses import asdict
import logging
from contextlib import contextmanager
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np
import qcodes as qc
from nanotune.device.device import Device
from nanotune.device.device_channel import DeviceChannel
from nanotune.device_tuner.tuningresult import MeasurementHistory
from nanotune.tuningstages.gatecharacterization1d import GateCharacterization1D
from nanotune.tuningstages.settings import (DataSettings, SetpointSettings,
    Classifiers)

logger = logging.getLogger(__name__)


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

    @setpoint_settings.setter
    def setpoint_settings(self, new_settings: SetpointSettings) -> None:
        self._setpoint_settings = new_settings
        self.metadata.update({'setpoint_settings': asdict(new_settings)})

    @property
    def data_settings(self) -> DataSettings:
        return self._data_settings

    @data_settings.setter
    def data_settings(self, new_settings: DataSettings) -> None:
        self._data_settings = new_settings
        self.metadata.update({'data_settings': asdict(new_settings)})

    def update_normalization_constants(self, device: Device):
        """
        Get the maximum and minimum signal measured of a device, for all
        readout methods specified in device.readout. It will first
        set all gate voltages to their respective most negative allowed values
        and then to their most positive allowed ones.

        Puts device back into state where it was before, gates are set in the
        order as they are set in device.gates.
        """
        available_readout = device.readout.available_readout()

        normalization_constants = {
            key: [0.0, 1.0] for key in available_readout.keys()
        }
        with set_back_voltages(device.gates):
            device.all_gates_to_lowest()
            for read_meth in available_readout.keys():
                val = float(device.readout[read_meth].get())
                normalization_constants[read_meth][0] = val

            device.all_gates_to_highest()
            for read_meth in available_readout.keys():
                val = float(device.readout[read_meth].get())
                normalization_constants[read_meth][1] = val
        device.normalization_constants(normalization_constants)

    def characterize_gates(
        self,
        device: Device,
        gates: List[DeviceChannel],
        use_safety_voltage_ranges: bool = False,
        comment: Optional[str] = None,
    ) -> MeasurementHistory:
        """
        Characterize multiple gates.
        It does not set any voltages.

        returns instance of MeasurementHistory
        """

        if comment is None:
            comment = f"Characterizing {gates}."
        if self.classifiers.pinchoff is None:
            raise KeyError("No pinchoff classifier found.")

        measurement_result = MeasurementHistory(device.name)
        updated_data_settings = self.measurement_data_settings(device)
        for gate in gates:
            if use_safety_voltage_ranges:
                v_range = gate.safety_voltage_range()
            else:
                v_range = device.current_valid_ranges()[gate.gate_id]

            setpoint_settings = self.measurement_setpoint_settings(
                [gate.voltage], [v_range], [gate.safety_voltage_range()]
            )

            stage = GateCharacterization1D(
                data_settings=updated_data_settings,
                setpoint_settings=setpoint_settings,
                readout=device.readout,
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

    def measure_initial_ranges_2D(
        self,
        device: Device,
        gate_to_set: DeviceChannel,
        gates_to_sweep: Sequence[DeviceChannel],
        voltage_step: float = 0.2,
    ) -> Tuple[Tuple[float, float], MeasurementHistory]:
        """
        Estimate the default voltage range to consider

        # Swap top_barrier and last barrier to pinch off agains it to
        # determine opposite corner of valid voltage space.
        Args:
            gate_to_set (nt.DeviceChannel):
            gates_to_sweep (list)
            voltage_step (flaot)
        Returns:
            tuple(float, float):
            MeasurementHistory:
        """
        if self.classifiers.pinchoff is None:
            raise KeyError("No pinchoff classifier.")

        device.all_gates_to_highest()
        data_settings = self.measurement_data_settings(device)
        measurement_result = MeasurementHistory(device.name)
        layout_ids = [g.gate_id for g in gates_to_sweep]
        skip_gates = dict.fromkeys(layout_ids, False)

        v_range = gate_to_set.safety_voltage_range()
        n_steps = int(abs(v_range[0] - v_range[1]) / voltage_step)

        v_steps = np.linspace(np.max(v_range), np.min(v_range), n_steps)
        for voltage in v_steps:
            gate_to_set.voltage(voltage)

            for gate in gates_to_sweep:
                if not skip_gates[gate.gate_id]:
                    setpoint_settings = self.measurement_setpoint_settings(
                        [gate.voltage],
                        [gate.safety_voltage_range()],
                        [gate.safety_voltage_range()]
                    )
                    stage = GateCharacterization1D(
                        data_settings=data_settings,
                        setpoint_settings=setpoint_settings,
                        readout=device.readout,
                        classifier=self.classifiers.pinchoff,
                    )
                    tuningresult = stage.run_stage()
                    tuningresult.status = device.get_gate_status()
                    measurement_result.add_result(
                        tuningresult,
                        f"characterization_{gate.name}",
                    )
                    if tuningresult.success:
                        skip_gates[gate.gate_id] = True
                        last_gate = gate

            if all(skip_gates.values()):
                break
        min_voltage = gate_to_set.voltage()

        # Swap top_barrier and last barrier to pinch off agains it to
        # determine opposite corner of valid voltage space.
        device.all_gates_to_highest()

        setpoint_settings = self.measurement_setpoint_settings(
            [gate_to_set.voltage],
            [gate.safety_voltage_range()], [gate.safety_voltage_range()]
        )

        v_range = last_gate.safety_voltage_range()
        n_steps = int(abs(v_range[0] - v_range[1]) / voltage_step)
        v_steps = np.linspace(v_range[1], v_range[0], n_steps)

        for voltage in v_steps:
            last_gate.voltage(voltage)
            stage = GateCharacterization1D(
                data_settings=self.data_settings,
                setpoint_settings=setpoint_settings,
                readout=device.readout,
                classifier=self.classifiers.pinchoff,
            )

            tuningresult = stage.run_stage()
            tuningresult.status = device.get_gate_status()
            measurement_result.add_result(
                tuningresult,
                f"characterization_{gate.name}",
            )
            if tuningresult.success:
                max_voltage = tuningresult.features["low_voltage"]
                break

        gate_to_set.parent.all_gates_to_highest()

        return (min_voltage, max_voltage), measurement_result

    def measurement_data_settings(
        self,
        device: Device,
    ) -> DataSettings:
        """Add device relevant readout settings.

        Returns:
            New data settings
        """
        new_datasettings = DataSettings(**asdict(self.data_settings))
        norm_csts = device.normalization_constants
        new_datasettings.normalization_constants = norm_csts
        return new_datasettings

    def measurement_setpoint_settings(
        self,
        parameters_to_sweep: Sequence[qc.Parameter],
        ranges_to_sweep: Sequence[Sequence[float]],
        safety_voltage_ranges: Sequence[Sequence[float]],
    ) -> SetpointSettings:
        """Add device relevant readout settings.

        Returns:
            New data settings
        """
        new_settings = SetpointSettings(**asdict(self.setpoint_settings))
        new_settings.parameters_to_sweep = parameters_to_sweep
        new_settings.ranges_to_sweep = ranges_to_sweep
        new_settings.safety_voltage_ranges = safety_voltage_ranges
        return new_settings
