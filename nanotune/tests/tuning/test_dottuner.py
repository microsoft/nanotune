# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest
from nanotune.device_tuner.tuner import TuningHistory
from nanotune.device_tuner.dottuner import (DotTuner, VoltageChangeDirection,
    DeviceState, check_new_voltage, RangeChangeSetting)
from nanotune.device_tuner.tuningresult import MeasurementHistory, TuningResult

def test_check_new_voltage(
    sim_device,
):
    sim_device.right_barrier.safety_voltage_range([-1, 0])
    new_direction = check_new_voltage(
        -1, sim_device.right_barrier, tolerance=0.1)
    assert new_direction == VoltageChangeDirection.negative

    sim_device.right_barrier.safety_voltage_range([-1, 0])
    new_direction = check_new_voltage(
        -0.01, sim_device.right_barrier, tolerance=0.1)
    assert new_direction == VoltageChangeDirection.positive


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
        target_state=DeviceState.doubledot,
    )
    assert sim_device.central_barrier.voltage() == -1.42947649216405
    gate_id = sim_device.central_barrier.gate_id
    assert sim_device.current_valid_ranges()[gate_id] == [
        -1.97565855285095, -0.974324774924975]

    dottuner.set_central_barrier(
        sim_device,
        target_state=DeviceState.singledot,
    )
    assert sim_device.central_barrier.voltage() == -0.974324774924975
    assert sim_device.current_valid_ranges()[gate_id] == [
        -1.97565855285095, -0.974324774924975]

    with pytest.raises(ValueError):
        dottuner.set_central_barrier(
        sim_device,
        target_state="doubledot",
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
        range_change_setting=RangeChangeSetting(
            relative_range_change=0.5,
            max_range_change=0.1,
            min_range_change=0.05
        )
    )
    assert new_v == -0.4

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.positive,
        range_change_setting=RangeChangeSetting(
            relative_range_change=0.5,
            max_range_change=1,
            min_range_change=0.05
        )
    )
    assert new_v == -0.25

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.positive,
        range_change_setting=RangeChangeSetting(
            relative_range_change=10,
            max_range_change=0.3,
            min_range_change=0.05
        )
    )
    assert new_v == -0.2

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.positive,
        range_change_setting=RangeChangeSetting(
            relative_range_change=0.0001,
            max_range_change=0.3,
            min_range_change=0.05
        )
    )
    assert new_v == -0.45


    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.negative,
        range_change_setting=RangeChangeSetting(
            relative_range_change=0.5,
            max_range_change=1,
            min_range_change=0.05
        )
    )
    assert new_v == -0.75

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.negative,
        range_change_setting=RangeChangeSetting(
            relative_range_change=10,
            max_range_change=3,
            min_range_change=0.05
        )
    )
    assert new_v == -3

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.negative,
        range_change_setting=RangeChangeSetting(
            relative_range_change=10,
            max_range_change=0.3,
            min_range_change=0.05
        )
    )
    assert new_v == -0.8

    new_v = dottuner.choose_new_gate_voltage(
        sim_device,
        gate_id=gate_id,
        voltage_change_direction=VoltageChangeDirection.negative,
        range_change_setting=RangeChangeSetting(
            relative_range_change=0.0001,
            max_range_change=0.3,
            min_range_change=0.05
        )
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
        sim_device, plunger_barrier_pairs=((2, 1)),
    )


def test_update_voltages_based_on_directives(
    dottuner,
    sim_device,
):
    barrier_changes = {
        1: VoltageChangeDirection.negative,
    }
    sim_device.gates[1].voltage(-0.5)
    new_direction = dottuner.update_voltages_based_on_directives(
        sim_device,
        barrier_changes,
        range_change_setting=RangeChangeSetting(
            relative_range_change=0.1,
            max_range_change=0.1,
        )
    )
    assert new_direction is None
    assert sim_device.gates[1].voltage() == -0.6


    barrier_changes = {
        1: VoltageChangeDirection.negative,
    }
    sim_device.gates[1].safety_voltage_range([-1, 0])
    sim_device.gates[1].voltage(-0.9)
    new_direction = dottuner.update_voltages_based_on_directives(
        sim_device,
        barrier_changes,
        range_change_setting=RangeChangeSetting(
            relative_range_change=0.1,
            max_range_change=0.2,
            min_range_change=0.1,
        )
    )

    assert new_direction == VoltageChangeDirection.negative
    assert sim_device.gates[1].voltage() == -0.9

    barrier_changes = {
        1: VoltageChangeDirection.positive,
    }

    sim_device.gates[1].safety_voltage_range([-1, 0])
    sim_device.gates[1].voltage(-0.1)
    new_direction = dottuner.update_voltages_based_on_directives(
        sim_device,
        barrier_changes,
        range_change_setting=RangeChangeSetting(
            relative_range_change=0.1,
            max_range_change=0.2,
            min_range_change=0.1,
        )
    )
    assert sim_device.gates[1].voltage() == -0.1
    assert new_direction == VoltageChangeDirection.positive


def test_set_central_barrier(
    dottuner,
    sim_device,
    sim_scenario_dottuning,
):
    sim_scenario_dottuning.run_next_step()

    sim_device.central_barrier.voltage(0)
    sim_device.central_barrier.safety_voltage_range([-3, 0])

    dottuner.set_central_barrier(
        sim_device,
        target_state = DeviceState.doubledot
    )
    assert sim_device.central_barrier.voltage() == -1.42947649216405

    sim_device.central_barrier.voltage(0)
    dottuner.set_central_barrier(
        sim_device,
        target_state = DeviceState.singledot
    )
    assert sim_device.central_barrier.voltage() == -0.974324774924975

    sim_device.central_barrier.voltage(0)
    dottuner.set_central_barrier(
        sim_device,
        target_state = DeviceState.pinchedoff
    )
    assert sim_device.central_barrier.voltage() == -1.97565855285095

    with pytest.raises(AssertionError):
        dottuner.set_central_barrier(
        sim_device,
        target_state = 'doubledot',
    )

    with pytest.raises(ValueError):
        dottuner.set_central_barrier(
        sim_device,
        target_state = DeviceState.undefined,
    )


def test_set_outer_barriers(
    dottuner,
    sim_device,
    sim_scenario_dottuning,
):
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    # do it twice to test loop without inserting
    # sim_scenario_dottuning.run_next_step() into measurement code
    new_direction = dottuner.set_outer_barriers(sim_device, gate_ids=[1, 1])

    assert sim_device.left_barrier.voltage() == -0.5141713904634877
    assert sim_device.current_valid_ranges()[1] == [
        -0.679226408802934, -0.44214738246082]
    assert sim_device.transition_voltages()[1] == -0.658219406468823
    assert new_direction is None

    new_direction = dottuner.set_outer_barriers(sim_device, gate_ids=1)
    assert sim_device.left_barrier.voltage() == -0.5141713904634877

    sim_device.left_barrier.voltage(0)
    new_direction = dottuner.set_outer_barriers(
        sim_device, gate_ids=[1], tolerance=2)
    assert new_direction == VoltageChangeDirection.positive
    assert sim_device.left_barrier.voltage() == 0


def test_adjust_all_barriers(dottuner, sim_device, sim_scenario_dottuning):
    sim_scenario_dottuning.run_next_step()
    sim_device.top_barrier.voltage(-0.8)
    sim_device.central_barrier.voltage(-0.6)
    voltage_change_direction= dottuner.adjust_all_barriers(
        sim_device,
        DeviceState.doubledot,
        VoltageChangeDirection.negative,
        helper_gate_id = 0,
        central_barrier_id = 3,
        outer_barriers_id= [],
    )
    assert voltage_change_direction is None
    assert sim_device.top_barrier.voltage() == -0.8500000000000001
    assert sim_device.central_barrier.voltage() == -1.42947649216405


def test_adjust_barriers_loop(
    dottuner,
    sim_device,
    sim_scenario_dottuning,
):
    sim_scenario_dottuning.run_next_step()
    sim_device.top_barrier.voltage(-0.8)
    sim_device.left_barrier.voltage(-0.6)
    sim_device.central_barrier.safety_voltage_range([-3, 0])
    sim_device.central_barrier.voltage(-0.95)

    dottuner.adjust_barriers_loop(
        sim_device,
        DeviceState.doubledot,
        initial_voltage_update={1: VoltageChangeDirection.negative},
        helper_gate_id = 0,
        central_barrier_id = 3,
        outer_barriers_id = [],
    )
    sim_device.left_barrier.voltage() == -0.65
    assert sim_device.central_barrier.voltage() == -0.95
    assert sim_device.top_barrier.voltage() == -0.8

    sim_device.left_barrier.safety_voltage_range([-1, 0])
    sim_device.left_barrier.voltage(-0.95)
    sim_device.central_barrier.voltage(-0.95)
    dottuner.adjust_barriers_loop(
        sim_device,
        DeviceState.doubledot,
        initial_voltage_update={2: VoltageChangeDirection.negative},
        helper_gate_id = 0,
        central_barrier_id = 3,
        outer_barriers_id= [],
    )
    # central_barrier is set by set_central_barrier based on a partial
    # measurement, which chooses -1 as it doesn't find a voltage at which
    # the signal is less than three quarter the max amplitude
    assert sim_device.central_barrier.voltage() == -1.42947649216405
    assert sim_device.left_barrier.voltage() == -0.95
    assert sim_device.top_barrier.voltage() == -0.75


def test_select_outer_barrier_directives(dottuner):
    barrier_directives = dottuner._select_outer_barrier_directives(
        termination_reasons= ['x more positive'],
    )
    assert barrier_directives[1] == VoltageChangeDirection.positive
    assert barrier_directives[4] == VoltageChangeDirection.positive

    barrier_directives = dottuner._select_outer_barrier_directives(
        termination_reasons= ['x more positive'], barrier_gate_ids=[1],
    )
    assert barrier_directives[1] == VoltageChangeDirection.positive
    assert len(barrier_directives) == 1

    with pytest.raises(ValueError):
        _ = dottuner._select_outer_barrier_directives(
            termination_reasons= ['x more psitive'],
        )


def test_update_dotregime_directive(
    dottuner,
):
    directive = dottuner._update_dotregime_directive(DeviceState.doubledot)
    assert directive[3] == VoltageChangeDirection.negative

    directive = dottuner._update_dotregime_directive(
        DeviceState.singledot, central_barrier_id=2)
    assert directive[2] == VoltageChangeDirection.positive

    with pytest.raises(ValueError):
        _ = dottuner._update_dotregime_directive(DeviceState.pinchedoff)


def test_update_gate_configuration(
    dottuner,
    sim_device,
    sim_scenario_dottuning
):
    sim_scenario_dottuning.run_next_step()
    last_result = TuningResult(stage='chargediagram', success=True)
    last_result.termination_reasons = []
    last_result.success = True

    sim_device.left_barrier.voltage(-0.3)
    sim_device.central_barrier.voltage(-0.4)
    sim_device.top_barrier.voltage(-0.8)

    dottuner.update_gate_configuration(
        sim_device,
        last_result,
        DeviceState.doubledot,
        helper_gate_id = 0,
        central_barrier_id = 3,
        outer_barrier_ids =[],
    )
    assert sim_device.left_barrier.voltage() == -0.3
    assert sim_device.central_barrier.voltage() == -0.45
    assert sim_device.top_barrier.voltage() == -0.8

    last_result.termination_reasons = ['x more negative']
    last_result.success = False
    sim_device.central_barrier.voltage(-0.4)
    sim_device.top_barrier.voltage(-0.8)

    dottuner.update_gate_configuration(
        sim_device,
        last_result,
        DeviceState.doubledot,
        helper_gate_id = 0,
        central_barrier_id = 3,
        outer_barrier_ids =[3],
    )
    assert sim_device.central_barrier.voltage() == -0.45
    assert sim_device.top_barrier.voltage() == -0.8

    sim_device.central_barrier.safety_voltage_range([-1, 0])
    sim_device.central_barrier.voltage(-0.95)

    sim_device.current_valid_ranges({0: [-1, 0]})
    sim_device.top_barrier.voltage(-0.8)

    dottuner.update_gate_configuration(
        sim_device,
        last_result,
        DeviceState.doubledot,
        helper_gate_id = 0,
        central_barrier_id = 3,
        outer_barrier_ids =[3],
    )
    # We sweep the central barrier instead of the outer barriers, so that
    # the central barrier is set as if it were an outer barrier. Hence
    # this value. It checks if it correctly launches adjust_all_barriers.
    assert sim_device.central_barrier.voltage() == -0.8962295628962297
    assert sim_device.top_barrier.voltage() == -0.8200000000000001

    last_result.termination_reasons = []
    last_result.success = False
    with pytest.raises(ValueError):
        dottuner.update_gate_configuration(
            sim_device,
            last_result,
            DeviceState.doubledot,
        )

