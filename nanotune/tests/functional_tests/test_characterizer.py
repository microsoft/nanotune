# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest
from nanotune.device_tuner.characterizer import Characterizer
from nanotune.device_tuner.tuningresult import MeasurementHistory
from nanotune.tuningstages.settings import Classifiers
from nanotune.device_tuner.tuner import set_back_voltages


def test_characterize_device_default(
    sim_device_playback,
    tuner_default_input,
    pinchoff_classifier,
    sim_scenario_device_characterization,
):
    self = Characterizer(**tuner_default_input)
    self.classifiers = Classifiers(pinchoff=pinchoff_classifier)
    gate_configurations = None
    device = sim_device_playback
    scenario = sim_scenario_device_characterization
    skip_gates = [device.top_barrier]

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

                scenario.run_next_step()
                sub_result = self.characterize_gates(
                    device,
                    gates=[gate],
                    use_safety_voltage_ranges=True,
                )
                measurement_result.add_result(sub_result)
                assert sub_result.success

    assert len(measurement_result.to_dict()) == 6

    self.close()
