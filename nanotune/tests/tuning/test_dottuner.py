# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest
from nanotune.device_tuner.dottuner import DotTuner
from nanotune.device_tuner.tuningresult import MeasurementHistory


def test_dottuner_init(
    tuner_default_input,
):
    tuner = DotTuner(**tuner_default_input)
    assert tuner._tuningresults_all == {}
    tuner.close()


def test_dottuner_helper_gate(
    dottuner,
    sim_device,
):
    sim_device.initial_valid_ranges(
        {0: [-0.6, -0.3]})  # top_barrier

    dottuner.set_helper_gate(
        sim_device,
        helper_gate_id=0,
        gates_for_init_range_measurement = [1, 2, 4],
    )
    assert sim_device.gates[0].voltage() == -0.3
    assert sim_device.current_valid_ranges()[0] == [-0.6, -0.3]

    def measure_initial_ranges_dummy(
        device,
        gate_to_set,
        gates_to_sweep,
        voltage_step  = 0.2,
    ):
        return [-0.4, -0.2], MeasurementHistory(device.name)

    sim_device.measure_initial_ranges_2D = measure_initial_ranges_dummy

    dottuner.set_helper_gate(
        sim_device,
        helper_gate_id=0,
        gates_for_init_range_measurement = [1, 2, 4],
    )
    assert len(dottuner._tuningresults_all[sim_device.name]) == 1
    assert sim_device.gates[0].voltage() == -0.2
    assert sim_device.current_valid_ranges()[0] == [-0.4, -0.2]

    sim_device._gates_dict = {}
    with pytest.raises(AssertionError):
        dottuner.set_helper_gate(sim_device, 0)