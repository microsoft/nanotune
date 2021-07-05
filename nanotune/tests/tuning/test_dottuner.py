# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from numpy import dot
import pytest
from nanotune.device.device import ReadoutMethods
from nanotune.device_tuner.tuner import TuningHistory
from nanotune.device_tuner.dottuner import (DotTuner, VoltageChangeDirection,
    DeviceState, check_readout_method)
from nanotune.device_tuner.tuningresult import MeasurementHistory, TuningResult


def test_check_readout_method(sim_device):
    check_readout_method(sim_device, ReadoutMethods.transport)

    with pytest.raises(ValueError):
        check_readout_method(sim_device, 'transport')

    with pytest.raises(ValueError):
        check_readout_method(sim_device, ReadoutMethods.sensing)


def test_dottuner_init(
    tuner_default_input,
):
    tuner = DotTuner(**tuner_default_input)
    assert tuner.tuning_history == TuningHistory()
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
        meas_res = MeasurementHistory(device.name)
        meas_res.add_result(TuningResult('init_ranges_2d', True))
        return [-0.4, -0.2], meas_res

    dottuner.measure_initial_ranges_2D = measure_initial_ranges_dummy
    sim_device.initial_valid_ranges(
        {0: sim_device.gates[0].safety_voltage_range()})

    dottuner.set_helper_gate(
        sim_device,
        helper_gate_id=0,
        gates_for_init_range_measurement = [1, 2, 4],
    )
    print(dottuner.tuning_history.results)
    assert len(dottuner.tuning_history.results[sim_device.name].tuningresults) == 1
    assert sim_device.gates[0].voltage() == -0.2
    assert sim_device.current_valid_ranges()[0] == [-0.4, -0.2]

    sim_device._gates_dict = {}
    with pytest.raises(KeyError):
        dottuner.set_helper_gate(sim_device, 0)


def test_set_central_barrier(
    sim_device,
    dottuner,
    sim_scenario_dottuning,
):
    sim_scenario_dottuning.run_next_step()
    dottuner.set_central_barrier(
        sim_device,
        desired_regime="doubledot",
    )
    assert sim_device.central_barrier.voltage() == -1.42947649216405
    gate_id = sim_device.central_barrier.gate_id
    assert sim_device.current_valid_ranges()[gate_id] == [
        -1.97565855285095, -0.974324774924975]

    dottuner.set_central_barrier(
        sim_device,
        desired_regime="singledot",
    )
    assert sim_device.central_barrier.voltage() == -0.974324774924975
    assert sim_device.current_valid_ranges()[gate_id] == [
        -1.97565855285095, -0.974324774924975]

    with pytest.raises(ValueError):
        dottuner.set_central_barrier(
        sim_device,
        desired_regime="dobledot",
    )


def test_choose_new_gate_voltage(
    sim_device,
    dottuner,
):
    gate_id = 2
    sim_device.gates[gate_id].voltage(-0.5)
    sim_device.gates[gate_id].safety_voltage_range([-3, 0])
    sim_device.current_valid_ranges({gate_id: [-1, 0]})

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.positive,
        relative_range_change = 0.5,
        max_range_change = 0.1,
        min_range_change= 0.05,
    )
    assert new_v == -0.4

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.positive,
        relative_range_change = 0.5,
        max_range_change = 1,
        min_range_change= 0.05,
    )
    assert new_v == -0.25

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.positive,
        relative_range_change = 10,
        max_range_change = 0.3,
        min_range_change= 0.05,
    )
    assert new_v == -0.2

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.positive,
        relative_range_change = 0.0001,
        max_range_change = 0.3,
        min_range_change= 0.05,
    )
    assert new_v == -0.45


    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.negative,
        relative_range_change = 0.5,
        max_range_change = 1,
        min_range_change= 0.05,
    )
    assert new_v == -0.75

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.negative,
        relative_range_change = 10,
        max_range_change = 3,
        min_range_change= 0.05,
    )
    assert new_v == -3

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.negative,
        relative_range_change = 10,
        max_range_change = 0.3,
        min_range_change= 0.05,
    )
    assert new_v == -0.8

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.negative,
        relative_range_change = 0.0001,
        max_range_change = 0.3,
        min_range_change= 0.05,
    )
    assert new_v == -0.55

    with pytest.raises(AssertionError):
        dottuner.choose_new_gate_voltage(
            sim_device,
            gate_id=2,
            voltage_change_direction='positive',
        )


def test_characterize_plunger(
    dottuner,
    sim_device,
    sim_scenario_dottuning,
):
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()

    new_range, device_state = dottuner.characterize_plunger(
        sim_device,
        sim_device.left_plunger,
    )
    assert new_range == (-0.331110370123374, -0.0710236745581861)
    assert device_state == DeviceState.undefined

    new_range, device_state = dottuner.characterize_plunger(
        sim_device,
        sim_device.left_plunger,
        noise_floor = 0.0002,  # min_signal = -0.11914071339135232
        # negative value below due to sim data interpolation
        open_signal = -0.2,  # 'max_signal': 0.3631990669884176
    )
    assert device_state == DeviceState.opencurrent

    new_range, device_state = dottuner.characterize_plunger(
        sim_device,
        sim_device.left_plunger,
        noise_floor = 0.4,  # min_signal = -0.11914071339135232
        open_signal = 0.6,  # 'max_signal': 0.3631990669884176
    )
    assert device_state == DeviceState.pinchedoff

    with pytest.raises(ValueError):
       _ = dottuner.characterize_plunger(
            sim_device,
            sim_device.left_plunger,
            main_readout_method='trnsport',
        )


def test_set_new_plunger_ranges(
    dottuner,
    sim_device,
    sim_scenario_dottuning,
):
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    success, barrier_changes = dottuner.set_new_plunger_ranges(
        sim_device,
        noise_floor = 0.02,
        open_signal = 0.1,
        plunger_barrier_pairs = [(2, 1), (2, 1)],
    )
    assert success
    assert not barrier_changes
    assert sim_device.current_valid_ranges()[2] == [
        -0.331110370123374, -0.0710236745581861]

    success, barrier_changes = dottuner.set_new_plunger_ranges(
        sim_device,
        noise_floor = 0.6,
        open_signal = 0.1,
        plunger_barrier_pairs = [(2, 1)],
    )
    assert not success
    assert barrier_changes[1] == VoltageChangeDirection.positive
    assert sim_device.current_valid_ranges()[2] == [
        -0.331110370123374, -0.0710236745581861]

    success, barrier_changes = dottuner.set_new_plunger_ranges(
        sim_device,
        noise_floor = 0.02,
        open_signal = -0.5,
        plunger_barrier_pairs = [(2, 1)],
    )
    assert not success
    assert barrier_changes[1] == VoltageChangeDirection.negative

    with pytest.raises(ValueError):
        _ = dottuner.set_new_plunger_ranges(
        sim_device, main_readout_method='trnsport',
    )

    with pytest.raises(ValueError):
        _ = dottuner.set_new_plunger_ranges(
        sim_device, plunger_barrier_pairs=((2, 1)),
    )


def test_update_barriers(
    dottuner,
    sim_device,
):
    barrier_changes = {
        1: VoltageChangeDirection.negative,
    }
    sim_device.gates[1].voltage(-0.5)
    new_direction = dottuner.update_barriers(
        sim_device,
        barrier_changes,
        relative_range_change = 0.1,
        max_range_change=0.1,
    )
    assert new_direction is None
    assert sim_device.gates[1].voltage() == -0.6


    barrier_changes = {
        1: VoltageChangeDirection.negative,
    }
    sim_device.gates[1].safety_voltage_range([-1, 0])
    sim_device.gates[1].voltage(-0.9)
    new_direction = dottuner.update_barriers(
        sim_device,
        barrier_changes,
        relative_range_change = 0.1,
        max_range_change=0.2,
        min_range_change=0.1,
    )

    assert new_direction == VoltageChangeDirection.negative
    assert sim_device.gates[1].voltage() == -0.9

    barrier_changes = {
        1: VoltageChangeDirection.positive,
    }

    sim_device.gates[1].safety_voltage_range([-1, 0])
    sim_device.gates[1].voltage(-0.1)
    new_direction = dottuner.update_barriers(
        sim_device,
        barrier_changes,
        relative_range_change = 0.1,
        max_range_change=0.2,
        min_range_change=0.1,
    )
    assert sim_device.gates[1].voltage() == -0.1
    assert new_direction == VoltageChangeDirection.positive


# def test_set_central_barrier(
#     dottuner,
#     sim_device,
# ):
#     dottuner.set_central_barrier(
#         sim_device,
#         desired_regime = DeviceState.doubledot
#     )






