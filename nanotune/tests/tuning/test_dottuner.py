# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from nanotune.tuningstages.settings import Classifiers
import pytest
import nanotune as nt
from nanotune.device_tuner.tuner import TuningHistory
from nanotune.device_tuner.dottuner import (DotTuner, VoltageChangeDirection,
    DeviceState, check_new_voltage, RangeChangeSetting)
from nanotune.device_tuner.tuningresult import MeasurementHistory, TuningResult
from nanotune.device.device_layout import DoubleDotLayout
from nanotune.device.device import NormalizationConstants
from qcodes.dataset.experiment_container import load_last_experiment


def test_device_state():
    assert DeviceState.pinchedoff.value == 0
    assert DeviceState.opencurrent.value == 1
    assert DeviceState.singledot.value == 2
    assert DeviceState.doubledot.value == 3
    assert DeviceState.undefined.value == 4


def test_voltage_direction_change():
    assert VoltageChangeDirection.positive.value == 0
    assert VoltageChangeDirection.negative.value == 1


def test_range_change_setting():
    setting = RangeChangeSetting()
    assert setting.relative_range_change == 0.1
    assert setting.max_range_change == 0.05
    assert setting.min_range_change == 0.01
    assert setting.tolerance == 0.1

    setting = RangeChangeSetting(0.2, 0.3, 0.4, 0.3)
    assert setting.relative_range_change == 0.2
    assert setting.max_range_change == 0.3
    assert setting.min_range_change == 0.4
    assert setting.tolerance == 0.3


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
    assert len(dottuner.tuning_history.results[sim_device.name].tuningresults) == 1
    assert sim_device.gates[0].voltage() == -0.2
    assert sim_device.current_valid_ranges()[0] == [-0.4, -0.2]

    sim_device._gates_dict = {}
    with pytest.raises(KeyError):
        dottuner.set_helper_gate(sim_device, 0)


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
    assert round(new_range[0], 2) == -0.33
    assert round(new_range[1], 2) == -0.07
    assert device_state == DeviceState.undefined
    dottuner.data_settings.noise_floor = 0.0002 # min_signal = -0.11914071339135232
    # negative value below due to sim data interpolation
    dottuner.data_settings.dot_signal_threshold = -0.2,  # 'max_signal': 0.3631990669884176
    new_range, device_state = dottuner.characterize_plunger(
        sim_device,
        sim_device.left_plunger,
    )
    assert device_state == DeviceState.opencurrent

    dottuner.data_settings.noise_floor = 0.4
    dottuner.data_settings.dot_signal_threshold = 0.6
    new_range, device_state = dottuner.characterize_plunger(
        sim_device,
        sim_device.left_plunger,
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
    dottuner.data_settings.noise_floor = 0.02
    dottuner.data_settings.dot_signal_threshold = 0.1

    barrier_changes = dottuner.set_new_plunger_ranges(
        sim_device,
        plunger_barrier_pairs = [(2, 1), (2, 1)],
    )
    assert not barrier_changes
    valid_range = sim_device.current_valid_ranges()[2]
    assert round(valid_range[0], 2) == -0.33
    assert round(valid_range[1], 2) == -0.07

    dottuner.data_settings.noise_floor = 0.6
    dottuner.data_settings.dot_signal_threshold = 0.1
    barrier_changes = dottuner.set_new_plunger_ranges(
        sim_device,
        plunger_barrier_pairs = [(2, 1)],
    )
    assert barrier_changes[1] == VoltageChangeDirection.positive
    valid_range = sim_device.current_valid_ranges()[2]
    assert round(valid_range[0], 2) == -0.33
    assert round(valid_range[1], 2) == -0.07

    dottuner.data_settings.noise_floor = 0.02
    dottuner.data_settings.dot_signal_threshold =-0.5
    barrier_changes = dottuner.set_new_plunger_ranges(
        sim_device,
        plunger_barrier_pairs = [(2, 1)],
    )
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
    assert round(sim_device.central_barrier.voltage(), 2) == -1.43
    gate_id = sim_device.central_barrier.gate_id

    valid_range = sim_device.current_valid_ranges()[gate_id]
    assert round(valid_range[0], 2) == -1.98
    assert round(valid_range[1], 2) == -0.97

    sim_device.central_barrier.voltage(0)
    dottuner.set_central_barrier(
        sim_device,
        target_state = DeviceState.singledot
    )
    assert round(sim_device.central_barrier.voltage(), 2) == -0.97
    valid_range = sim_device.current_valid_ranges()[gate_id]
    assert round(valid_range[0], 2) == -1.98
    assert round(valid_range[1], 2) == -0.97

    sim_device.central_barrier.voltage(0)
    dottuner.set_central_barrier(
        sim_device,
        target_state = DeviceState.pinchedoff
    )
    assert round(sim_device.central_barrier.voltage(), 2) == -1.98

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

    assert round(sim_device.left_barrier.voltage(), 2) == -0.51
    valid_range = sim_device.current_valid_ranges()[1]
    assert round(valid_range[0], 2) == -0.68
    assert round(valid_range[1], 2) == -0.44

    assert round(sim_device.transition_voltages()[1], 2) == -0.66
    assert new_direction is None

    new_direction = dottuner.set_outer_barriers(sim_device, gate_ids=1)
    assert round(sim_device.left_barrier.voltage(), 2) == -0.51

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
    assert round(sim_device.top_barrier.voltage(), 2) == -0.85
    assert round(sim_device.central_barrier.voltage(), 2) == -1.43


def test_adjust_all_barriers_loop(
    dottuner,
    sim_device,
    sim_scenario_dottuning,
):
    sim_scenario_dottuning.run_next_step()
    sim_device.top_barrier.voltage(-0.8)
    sim_device.left_barrier.voltage(-0.6)
    sim_device.central_barrier.safety_voltage_range([-3, 0])
    sim_device.central_barrier.voltage(-0.95)

    dottuner.adjust_all_barriers_loop(
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
    dottuner.adjust_all_barriers_loop(
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
    assert round(sim_device.central_barrier.voltage(), 2) == -1.43
    assert round(sim_device.left_barrier.voltage(), 2) == -0.95
    assert round(sim_device.top_barrier.voltage(), 2) == -0.75


def test_select_outer_barrier_directives(dottuner):
    barrier_directives = dottuner._select_outer_barrier_directives(
        termination_reasons= ['x more positive'],
    )
    assert barrier_directives[1] == VoltageChangeDirection.positive
    assert barrier_directives[5] == VoltageChangeDirection.positive

    barrier_directives = dottuner._select_outer_barrier_directives(
        termination_reasons= ['x more positive'], outer_barriers_id=[1],
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
    assert round(sim_device.central_barrier.voltage(), 2) == -0.9
    assert round(sim_device.top_barrier.voltage(), 2) == -0.82

    last_result.termination_reasons = []
    last_result.success = False
    with pytest.raises(ValueError):
        dottuner.update_gate_configuration(
            sim_device,
            last_result,
            DeviceState.doubledot,
        )

def set_central_and_outer_barriers(dottuner, sim_device):
    sim_device.top_barrier.voltage(-0.8)
    sim_device.left_barrier.voltage(-0.6)
    sim_device.right_barrier.voltage(-0.5)
    sim_device.current_valid_ranges({0: [-1, 0]})

    def set_central_barrier_dummy(device, target_state, gate_id):
        device.central_barrier.voltage(-0.57)

    def set_outer_barriers_dummy(device, gate_ids):
        new_val = device.left_barrier.voltage() - 0.1
        device.left_barrier.voltage(new_val)
        new_val = device.right_barrier.voltage() - 0.1
        device.right_barrier.voltage(new_val)
        if device.top_barrier.voltage() > -0.87:
            return VoltageChangeDirection.negative
        else:
            return None

    dottuner.set_central_barrier = set_central_barrier_dummy
    dottuner.set_outer_barriers = set_outer_barriers_dummy
    dottuner.set_central_and_outer_barriers(
        sim_device,
        device_layout=DoubleDotLayout,
        target_state=DeviceState.doubledot,
    )

    assert round(sim_device.top_barrier.voltage(), 2) == -0.88
    assert round(sim_device.left_barrier.voltage(), 2) == -1.2
    assert round(sim_device.right_barrier.voltage(), 2) == -1.1
    assert round(sim_device.central_barrier.voltage(), 2) == -0.57


def test_adjust_outer_barriers_possibly_helper_gate(dottuner, sim_device):

    sim_device.current_valid_ranges({0: [-1, 0]})
    sim_device.gates[1].voltage(-0.95)
    sim_device.gates[1].safety_voltage_range([-1, 0])

    sim_device.top_barrier.voltage(-0.8)
    sim_device.right_barrier.voltage(-0.6)

    def set_outer_barriers_dummy(device, gate_ids):
        new_val = device.left_barrier.voltage() + 0.1
        device.left_barrier.voltage(new_val)
        new_val = device.right_barrier.voltage() + 0.1
        device.right_barrier.voltage(new_val)
        if device.top_barrier.voltage() > -0.87:
            return VoltageChangeDirection.negative
        else:
            return None

    barrier_changes = {1: VoltageChangeDirection.negative}
    dottuner.set_outer_barriers = set_outer_barriers_dummy

    dottuner.adjust_outer_barriers_possibly_helper_gate(
        sim_device,
        DoubleDotLayout,
        barrier_changes,
    )

    assert round(sim_device.top_barrier.voltage(), 2) == -0.88
    assert round(sim_device.left_barrier.voltage(), 2)  == -0.45
    assert round(sim_device.right_barrier.voltage(), 2)  == -0.1


def test_set_valid_plunger_ranges(dottuner, sim_device):

    def set_new_plunger_ranges_dummy(
        device, plunger_barrier_pairs,
    ):
        device.current_valid_ranges(
        {
            device.left_plunger.gate_id: (-0.2, 0),
            device.right_plunger.gate_id: (-0.2, 0),
        })
        if device.left_barrier.voltage() < -0.6:
            barrier_changes = None
        else:
            barrier_changes = {
                device.left_barrier.gate_id: VoltageChangeDirection.negative,
                device.right_barrier.gate_id: VoltageChangeDirection.negative,
            }
        return barrier_changes

    def adjust_outer_barriers_possibly_helper_gate_dummy(
        device, device_layout, barrier_changes,
    ):
        curr_val = device.top_barrier.voltage()
        device.top_barrier.voltage(curr_val - 0.05)

        curr_val = device.left_barrier.voltage()
        device.left_barrier.voltage(curr_val - 0.05)

        curr_val = device.right_barrier.voltage()
        device.right_barrier.voltage(curr_val - 0.05)

    sim_device.top_barrier.voltage(-0.42)
    sim_device.left_barrier.voltage(-0.4)
    sim_device.right_barrier.voltage(-0.45)

    dottuner.set_new_plunger_ranges = set_new_plunger_ranges_dummy
    dottuner.adjust_outer_barriers_possibly_helper_gate = adjust_outer_barriers_possibly_helper_gate_dummy
    dottuner.data_settings.noise_floor = 0.02
    dottuner.data_settings.dot_signal_threshold = 0.1

    dottuner.set_valid_plunger_ranges(
        sim_device,
        DoubleDotLayout,
        # noise_floor= 0.02,
        # dot_signal_threshold=0.1,
    )
    assert round(sim_device.top_barrier.voltage(), 2)  == -0.62
    assert round(sim_device.left_barrier.voltage(), 2)  == -0.6
    assert round(sim_device.right_barrier.voltage(), 2)  == -0.65


def test_take_high_res_dot_segments(
    dottuner, sim_device, sim_scenario_dottuning,
):
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    tuningresult = TuningResult('chargediagram', success=True)
    tuningresult.ml_result = {
        'dot_segments': {
            1: {'voltage_ranges':
            [(-0.3, -0.171428571428571), (-0.3, -0.171428571428571)],
            'predicted_regime': 3},
            2: {'voltage_ranges':
            [(-0.3, -0.171428571428571), (-0.15, 0.0)],
            'predicted_regime': 3},
            }
    }
    voltage_precision = 0.01
    dottuner.take_high_res_dot_segments(
        sim_device,
        dot_segments=tuningresult.ml_result['dot_segments'],
        gate_ids=DoubleDotLayout.plungers(),
        target_state=DeviceState.doubledot,
        voltage_precision=voltage_precision,
    )
    nt.set_database(
        dottuner.data_settings.db_name,
        dottuner.data_settings.db_folder,
    )
    exp = load_last_experiment()
    assert exp.last_counter == 2

    ds = nt.Dataset(
        1,
        dottuner.data_settings.db_name,
        dottuner.data_settings.db_folder
    )
    n_pt = abs(ds.data.voltage_x.values[0] - ds.data.voltage_x.values[-1])
    n_pt = int(n_pt/voltage_precision)
    assert n_pt == 12

def test_take_high_resolution_diagram(
    dottuner, sim_device, sim_scenario_dottuning,
):
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()

    sim_device.current_valid_ranges(
        {
            sim_device.left_plunger.gate_id: [-0.3, 0],
            sim_device.right_plunger.gate_id: [-0.3, 0],
        }
    )

    dottuner.data_settings.segment_size = 0.1
    voltage_precisions = (0.02, 0.01)
    dottuner.take_high_resolution_diagram(
        sim_device,
        gate_ids=DoubleDotLayout.plungers(),
        target_state=DeviceState.doubledot,
        take_segments=True,
        voltage_precisions=voltage_precisions,
    )

    nt.set_database(
        dottuner.data_settings.db_name,
        dottuner.data_settings.db_folder,
    )
    exp = load_last_experiment()
    assert exp.last_counter == 5

    ds = nt.Dataset(
        1,
        dottuner.data_settings.db_name,
        dottuner.data_settings.db_folder
    )
    n_pt = abs(ds.data.voltage_x.values[0] - ds.data.voltage_x.values[-1])
    n_pt = int(n_pt/voltage_precisions[0])
    assert n_pt == 15

    nt.set_database(
        dottuner.data_settings.segment_db_name,
        dottuner.data_settings.segment_db_folder,
    )
    exp = load_last_experiment()
    assert exp.last_counter == 8


def test_tune_dot_regime(dottuner, sim_device):

    def set_valid_plunger_ranges_dummy(device, device_layout):
        device.current_valid_ranges({
            device_layout.plungers()[0]: (-0.3, 0),
            device_layout.plungers()[1]: (-0.3, 0),
        })

    def get_charge_diagram_dummy(
        device, gates_to_sweep, use_safety_voltage_ranges = False,
        iterate = False, voltage_precision = None,
    ):
        if device.right_barrier.voltage() < -0.6:
            tuning_result = TuningResult('chargediagram', success=True)
        else:
            tuning_result = TuningResult('chargediagram', success=False)
        tuning_result.ml_result = {
            'dot_segments': {
                1: {'voltage_ranges':
                [(-0.3, -0.171428571428571), (-0.3, -0.171428571428571)],
                'predicted_regime': 3},
                2: {'voltage_ranges':
                [(-0.3, -0.171428571428571), (-0.15, 0.0)],
                'predicted_regime': 3},
                }
        }
        return tuning_result

    def update_gate_configuration_dummy(device, tuningresult, target_state):
        curr_val = device.right_barrier.voltage()
        device.right_barrier.voltage(curr_val - 0.1)

    def set_central_and_outer_barriers_dummy(
        device, device_layout, target_state
    ):
        assert target_state == DeviceState.doubledot
        device.top_barrier.voltage() == -0.6

        device.gates[device_layout.central_barrier()].voltage(-0.44)
        device.gates[device_layout.outer_barriers()[0]].voltage(-0.43)

    dottuner.set_central_and_outer_barriers = set_central_and_outer_barriers_dummy
    dottuner.set_valid_plunger_ranges = set_valid_plunger_ranges_dummy
    dottuner.get_charge_diagram = get_charge_diagram_dummy
    dottuner.update_gate_configuration = update_gate_configuration_dummy

    sim_device.right_barrier.voltage(-0.4)
    success = dottuner.tune_dot_regime(
        sim_device,
        DoubleDotLayout,
        DeviceState.doubledot,
        take_high_res=True,
        continue_tuning=False,
        max_iter=10,
    )
    assert success
    assert sim_device.current_valid_ranges()[DoubleDotLayout.plungers()[0]] == [-0.3, -0.171428571428571]
    assert sim_device.current_valid_ranges()[DoubleDotLayout.plungers()[1]] == [-0.15, 0.0]

    assert sim_device.left_barrier.voltage() == -0.43
    assert sim_device.right_barrier.voltage() == -0.7
    assert sim_device.central_barrier.voltage() == -0.44

    sim_device.right_barrier.voltage(-0.4)
    success = dottuner.tune_dot_regime(
        sim_device,
        DoubleDotLayout,
        DeviceState.doubledot,
        take_high_res=True,
        continue_tuning=False,
        max_iter=2,
    )
    assert not success
    assert sim_device.current_valid_ranges()[DoubleDotLayout.plungers()[0]] == [-0.3, 0]
    assert sim_device.current_valid_ranges()[DoubleDotLayout.plungers()[1]] == [-0.3, 0]

    sim_device.right_barrier.voltage(0)
    success = dottuner.tune_dot_regime(
        sim_device,
        DoubleDotLayout,
        DeviceState.doubledot,
        take_high_res=True,
        continue_tuning=True,
        max_iter=4,
    )
    assert not success


def test_tune(dottuner, sim_device):

    dottuner.tuning_history.update(
        sim_device.name, TuningResult('tuning', success=True)
    )
    def update_normalization_constants_dummy(device):
        for gate in device.gates:
            assert gate.voltage() == 0
        device.normalization_constants = NormalizationConstants(
            transport=(0.1, 0.9)
        )
    def set_helper_gate_dummy(
        device, helper_gate_id, gates_for_init_range_measurement
    ):
        device.gates[helper_gate_id].voltage(-0.6)
        assert gates_for_init_range_measurement == DoubleDotLayout.barriers()
        assert device.normalization_constants == NormalizationConstants(
            transport=(0.1, 0.9)
        )

    def tune_dot_regime_dummy(
        device, device_layout, target_state, max_iter, take_high_res,
        continue_tuning):
        assert sim_device.top_barrier.voltage() == -0.6

        assert target_state == DeviceState.doubledot
        assert max_iter == 10
        assert take_high_res
        assert not continue_tuning
        return True

    dottuner.tune_dot_regime = tune_dot_regime_dummy
    dottuner.set_helper_gate = set_helper_gate_dummy
    dottuner.update_normalization_constants = update_normalization_constants_dummy

    sim_device.all_gates_to_lowest()
    success, tuning_history = dottuner.tune(
            sim_device,
            DoubleDotLayout,
            DeviceState.doubledot,
            take_high_res=True,
            continue_tuning=False,
            max_iter=10,
        )

    assert success
    assert tuning_history.tuningresults['tuning'] == TuningResult('tuning', success=True)

    dottuner.classifiers = Classifiers()
    with pytest.raises(ValueError):
        dottuner.tune(
            sim_device,
            DoubleDotLayout,
            DeviceState.doubledot,
            take_high_res=True,
            continue_tuning=True,
            max_iter=4,
        )
