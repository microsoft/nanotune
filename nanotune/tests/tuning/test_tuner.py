import copy

import pytest
import matplotlib.pyplot as plt
from dataclasses import asdict
from nanotune.device_tuner.tuner import set_back_voltages, linear_voltage_steps
from nanotune.device.device import NormalizationConstants
from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.tuningstages.settings import Classifiers, DataSettings

atol = 1e-05


def test_set_back_voltages(gate_1, gate_2):
    gate_1.voltage(-0.8)
    gate_2.voltage(-0.9)
    assert gate_1.voltage() == -0.8
    assert gate_2.voltage() == -0.9

    with set_back_voltages([gate_1, gate_2]):
        assert gate_1.voltage() == -0.8
        assert gate_2.voltage() == -0.9
        gate_1.voltage(-0.5)
        gate_2.voltage(-0.4)
        assert gate_1.voltage() == -0.5
        assert gate_2.voltage() == -0.4

    assert gate_1.voltage() == -0.8
    assert gate_2.voltage() == -0.9


def test_tuner_init_and_attributes(tuner, tmp_path):
    assert tuner.data_settings.db_name == "temp.db"
    assert tuner.data_settings.db_folder == str(tmp_path)
    assert tuner.data_settings.experiment_id == None

    new_data_settings = DataSettings(db_name="other_temp.db")
    data_settings = DataSettings(**asdict(tuner.data_settings))

    tuner.data_settings = new_data_settings
    data_settings.update(new_data_settings)
    assert tuner.data_settings == data_settings

    assert tuner.setpoint_settings.voltage_precision == 0.001


def test_update_normalization_constants(
    tuner,
    sim_device_pinchoff,
):
    previous_constants = sim_device_pinchoff.normalization_constants
    sim_device_pinchoff.normalization_constants = NormalizationConstants()

    tuner.update_normalization_constants(sim_device_pinchoff)
    updated_constants = sim_device_pinchoff.normalization_constants

    assert updated_constants.transport == previous_constants.transport
    assert updated_constants.sensing == (0., 1.)
    assert updated_constants.rf == (0., 1.)


def test_characterize_gates(
    tuner,
    sim_device_pinchoff,
    pinchoff_classifier,
):
    tuner.classifiers = Classifiers(pinchoff=pinchoff_classifier)

    result = tuner.characterize_gates(
        sim_device_pinchoff,
        [sim_device_pinchoff.right_plunger],
        use_safety_voltage_ranges=True,
    )
    plt.close('all')

    stage_name = "characterization_" + sim_device_pinchoff.right_plunger.name
    assert isinstance(result, TuningResult)
    assert result.status == sim_device_pinchoff.get_gate_status()
    comment = f"Characterizing {[sim_device_pinchoff.right_plunger]}."
    assert result.comment == comment


    sim_device_pinchoff.current_valid_ranges(
        {sim_device_pinchoff.right_plunger.gate_id: (-0.2, 0)}
    )
    result = tuner.characterize_gates(
        sim_device_pinchoff,
        [sim_device_pinchoff.right_plunger],
        use_safety_voltage_ranges=False,
        comment = "first run",
        iterate=False,
    )
    assert result.comment == "first run"
    assert not result.success

    result = tuner.characterize_gates(
        sim_device_pinchoff,
        [sim_device_pinchoff.right_plunger],
        use_safety_voltage_ranges=False,
        iterate=True,
    )
    assert result.success
    assert len(result.data_ids) == 2

    with pytest.raises(KeyError):
        tuner.classifiers = Classifiers()
        _ = tuner.characterize_gates(
            sim_device_pinchoff,
            [sim_device_pinchoff.right_plunger],
            use_safety_voltage_ranges=True,
        )


def test_measurement_data_settings(
    tuner,
    sim_device_pinchoff,
):
    prev_settings = copy.deepcopy(tuner.data_settings)
    new_data_settings = tuner.measurement_data_settings(sim_device_pinchoff)
    assert prev_settings.normalization_constants != new_data_settings.normalization_constants
    assert new_data_settings.normalization_constants == sim_device_pinchoff.normalization_constants


def test_measurement_setpoint_settings(
    tuner,
    sim_device_pinchoff,
):
    prev_settings = copy.deepcopy(tuner.setpoint_settings)
    new_settings = tuner.measurement_setpoint_settings(
        [sim_device_pinchoff.left_barrier],
        [[-2, 0]],
        [[-3, 0]])
    assert prev_settings != new_settings

    assert new_settings.parameters_to_sweep == [sim_device_pinchoff.left_barrier]
    assert new_settings.ranges_to_sweep == [[-2, 0]]
    assert new_settings.safety_voltage_ranges == [[-3, 0]]


def test_get_pairwise_pinchoff(
    tuner,
    sim_device_gatecharacterization2d,
    pinchoff_classifier,
):
    tuner.classifiers = Classifiers(pinchoff=pinchoff_classifier)
    gate_to_set = sim_device_gatecharacterization2d.left_plunger
    gates_to_sweep = [sim_device_gatecharacterization2d.right_plunger]

    v_steps = linear_voltage_steps(
            [0, -2], 0.2)
    (measurement_result,
     last_gate_to_pinchoff,
     last_voltage) = tuner.get_pairwise_pinchoff(
        sim_device_gatecharacterization2d,
        gate_to_set,
        gates_to_sweep,
        v_steps,
    )
    assert last_voltage == -0.44444444444444464
    assert len(measurement_result.to_dict()) == 9
    assert last_gate_to_pinchoff.full_name == gates_to_sweep[0].full_name




