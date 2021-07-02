import logging
from nanotune.device.device_channel import DeviceChannel
from typing import Dict, Optional, Sequence

from nanotune.device.device import Device
from nanotune.device_tuner.tuner import (Tuner, set_back_voltages,
    DataSettings, SetpointSettings, Classifiers)
from nanotune.device_tuner.tuningresult import MeasurementHistory
logger = logging.getLogger(__name__)


class Characterizer(Tuner):
    """
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
        skip_gates: Optional[Sequence[DeviceChannel]],
        gate_configurations: Optional[Dict[int, Dict[int, float]]] = None,
    ) -> MeasurementHistory:
        """
        gate_configurations: Dict[int, Dict[int, float]]; with gate
        configuration to be applied for individual gate characterizations.
        Example: Set top barrier of a 2DEG device.
        """
        if gate_configurations is None:
            gate_configurations = {}

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
