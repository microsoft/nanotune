import logging
from typing import Dict, Optional

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
        gate_configurations: Optional[Dict[int, Dict[int, float]]] = None,
    ) -> MeasurementHistory:
        """
        gate_configurations: Dict[int, Dict[int, float]]; with gate
        configuration to be applied for individual gate characterizations.
        Example: Set top barrier of a 2DEG device.
        """
        if self.qcodes_experiment.sample_name != Device.name:
            logger.warning(
                (
                    "The device's name does match the"
                    " the sample name in qcodes experiment."
                )
            )
        if gate_configurations is None:
            gate_configurations = {}

        measurement_result = MeasurementHistory(device.name)

        for gate in device.gates:
            with set_back_voltages(device.gates):
                gate_id = gate.layout_id()
                if gate_id in gate_configurations.keys():
                    gate_conf = gate_configurations[gate_id].items()
                    for other_id, voltage in gate_conf:
                        device.gates[other_id].voltage(voltage)

                sub_result = self.characterize_gates(
                    device,
                    gates=device.gates,
                    use_safety_ranges=True,
                )
                measurement_result.update(sub_result)

        return measurement_result
