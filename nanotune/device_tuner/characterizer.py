# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import logging
from nanotune.device.device_channel import DeviceChannel
from typing import Dict, Optional, Sequence

from nanotune.device.device import Device
from nanotune.device_tuner.tuner import (Tuner, set_back_voltages,
    DataSettings, SetpointSettings, Classifiers)
from nanotune.device_tuner.tuningresult import MeasurementHistory
logger = logging.getLogger(__name__)


class Characterizer(Tuner):
    """Tuner sub-class specializing on device characterization.

    Attributes:
        classifiers (Classifiers): a setting.Classifiers instance
            holding all required classifiers. Eg. pinchoff.
        data_settings (DataSettings): A settings.DataSettings instance with
            data related information such as `db_name` and
            `normalization_constants'.
        setpoint_settings (SetpointSettings): A settings.SetpointSettings
            instance with setpoint related information such as
            `voltage_precision`.
        tuning_history (TuningHistory): A TuningHistory instance holding all
            tuning results.
    """
    def __init__(
        self,
        name: str,
        data_settings: DataSettings,
        classifiers: Classifiers,
        setpoint_settings: SetpointSettings,
    ) -> None:
        super().__init__(
            name,
            data_settings,
            classifiers,
            setpoint_settings,
        )

    def characterize(
        self,
        device: Device,
        skip_gates: Optional[Sequence[DeviceChannel]] = None,
        gate_configurations: Optional[Dict[int, Dict[int, float]]] = None,
    ) -> MeasurementHistory:
        """Characterizes a device by characterizing each gate individually.
        Specific gates can be skipped, eg. the top barrier of a 2DEG device.

        Args:
            device (nt.Device): device to tune.
            skip_gates (Sequence[DeviceChannel]): optional list of gates which
                should not be characterized.
            gate_configurations (Dict[int, Dict[int, float]]): optional gate
                voltage combinations at which gates should be characterized.
                Maps gate IDs of gates to characteris onto dictionaries, which
                in turn map gate IDs of gates to set to their respective
                voltages.

        Returns:
            MeasurementHistory: Collection of all tuning results.
        """
        if gate_configurations is None:
            gate_configurations = {}
        if skip_gates is None:
            skip_gates = []

        measurement_result = MeasurementHistory(device.name)

        for gate in device.gates:
            if gate not in skip_gates:
                with set_back_voltages(device.gates):
                    gate_id = gate.gate_id
                    if gate_id in gate_configurations.keys():
                        gate_conf = gate_configurations[gate_id]
                        for other_id, voltage in gate_conf.items():
                            device.gates[other_id].voltage(voltage)

                    sub_result = self.characterize_gate(
                        device,
                        gate,
                        use_safety_voltage_ranges=True,
                    )
                    measurement_result.add_result(sub_result)

        return measurement_result
