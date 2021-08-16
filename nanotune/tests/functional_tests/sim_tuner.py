# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import List, Optional
from nanotune.device_tuner.dottuner import DotTuner
from nanotune.device_tuner.tuner import (DataSettings, SetpointSettings,
    Classifiers)
from nanotune.device.device import Device
from nanotune.device.device_channel import DeviceChannel
from sim.simulation_scenario import SimulationScenario
from nanotune.device_tuner.tuningresult import TuningResult


class SimDotTuner(DotTuner):
    def __init__(
        self,
        name: str,
        data_settings: DataSettings,
        classifiers: Classifiers,
        setpoint_settings: SetpointSettings,
        sim_scenario: SimulationScenario,
    ) -> None:
        super().__init__(
            name,
            data_settings,
            classifiers,
            setpoint_settings,
        )
        self.sim_scenario = sim_scenario

    def get_charge_diagram(
        self,
        device: Device,
        gates_to_sweep: List[DeviceChannel],
        use_safety_voltage_ranges: bool = False,
        iterate: bool = False,
        voltage_precision: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> TuningResult:
        """ """
        self.sim_scenario.run_next_step()
        tuningresult = super().get_charge_diagram(
            device, gates_to_sweep, use_safety_voltage_ranges, iterate,
            voltage_precision, comment,
        )
        return tuningresult

    def characterize_gate(
        self,
        device: Device,
        gate: DeviceChannel,
        use_safety_voltage_ranges: bool = False,
        iterate: bool = False,
        voltage_precision: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> TuningResult:
        """ """
        self.sim_scenario.run_next_step()
        tuningresult = super().characterize_gate(
            device, gate, use_safety_voltage_ranges, iterate,
            voltage_precision, comment,
        )
        return tuningresult
