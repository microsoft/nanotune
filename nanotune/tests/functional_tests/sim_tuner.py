# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import Dict, List, Optional, Sequence, Tuple, Any
from nanotune.device_tuner.dottuner import (DotTuner, VoltageChangeDirection,
    DeviceState, check_new_voltage, RangeChangeSetting)
from nanotune.device_tuner.tuner import (DataSettings, SetpointSettings,
    Classifiers)
from nanotune.device.device import Device
from nanotune.device.device_channel import DeviceChannel
from sim.simulation_scenario import SimulationScenario
from nanotune.device_tuner.tuningresult import MeasurementHistory, TuningResult


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
        print('next sim scenario charge diagram')
        print(f'sweep {[g.label for g in gates_to_sweep]}')
        print(f'gate status {device.get_gate_status()}')
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
        print('next sim scenario gate characterization')
        print(f'characterize {gate.label}')
        print(f'gate status {device.get_gate_status()}')
        tuningresult = super().characterize_gate(
            device, gate, use_safety_voltage_ranges, iterate,
            voltage_precision, comment,
        )
        return tuningresult
