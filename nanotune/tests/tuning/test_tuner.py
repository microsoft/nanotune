import copy
from nanotune.tests.mock_classifier import MockClassifer

import pytest
import matplotlib.pyplot as plt
from dataclasses import asdict
import nanotune as nt
from nanotune.device_tuner.tuner import (TuningHistory, set_back_voltages,
    linear_voltage_steps)
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

    assert tuner.tuning_history == TuningHistory()

    new_data_settings = DataSettings(db_name="other_temp.db")
    data_settings = DataSettings(**asdict(tuner.data_settings))

    tuner.data_settings = new_data_settings
    data_settings.update(new_data_settings)
    assert tuner.data_settings == data_settings

    assert tuner.setpoint_settings.voltage_precision == 0.001


def test_update_normalization_constants(
    tuner,
    sim_device,
    sim_scenario_device_characterization
):
    sim_scenario_device_characterization.run_next_step()
    previous_constants = sim_device.normalization_constants
    sim_device.normalization_constants = NormalizationConstants()

    tuner.update_normalization_constants(sim_device)
    updated_constants = sim_device.normalization_constants

    assert updated_constants.transport == previous_constants.transport
    assert updated_constants.sensing == (0., 1.)
    assert updated_constants.rf == (0., 1.)


def test_characterize_gate(
    tuner,
    sim_device,
    sim_scenario_device_characterization,
    pinchoff_classifier,
):
    sim_scenario_device_characterization.run_next_step()
    tuner.classifiers = Classifiers(pinchoff=pinchoff_classifier)

    result = tuner.characterize_gate(
        sim_device,
        sim_device.left_barrier,
        use_safety_voltage_ranges=True,
    )
    assert isinstance(result, TuningResult)
    assert result.status == sim_device.get_gate_status()
    comment = f"Characterizing {sim_device.left_barrier}."
    assert result.comment == comment

    assert len(tuner.tuning_history.results[sim_device.name].tuningresults) == 1


    sim_device.current_valid_ranges(
        {sim_device.left_barrier.gate_id: (-0.2, 0)}
    )
    result = tuner.characterize_gate(
        sim_device,
        sim_device.left_barrier,
        use_safety_voltage_ranges=False,
        comment = "first run",
        iterate=False,
    )
    assert result.comment == "first run"
    assert not result.success

    result = tuner.characterize_gate(
        sim_device,
        sim_device.left_barrier,
        use_safety_voltage_ranges=False,
        iterate=True,
    )
    assert result.success
    assert len(result.data_ids) == 2

    with pytest.raises(KeyError):
        tuner.classifiers = Classifiers()
        _ = tuner.characterize_gate(
            sim_device,
            sim_device.left_barrier,
            use_safety_voltage_ranges=True,
        )


def test_measurement_data_settings(
    tuner,
    sim_device,
):
    prev_settings = copy.deepcopy(tuner.data_settings)
    new_data_settings = tuner.measurement_data_settings(sim_device)
    assert prev_settings.normalization_constants != new_data_settings.normalization_constants
    assert new_data_settings.normalization_constants == sim_device.normalization_constants


def test_measurement_setpoint_settings(
    tuner,
    sim_device,
):
    prev_settings = copy.deepcopy(tuner.setpoint_settings)
    new_settings = tuner.measurement_setpoint_settings(
        [sim_device.left_barrier],
        [[-2, 0]],
        [[-3, 0]])
    assert prev_settings != new_settings

    assert new_settings.parameters_to_sweep == [sim_device.left_barrier]
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


def test_get_chargediagram(
    tuner,
    sim_device,
    sim_scenario_dottuning,
    tmp_path,
):
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    sim_scenario_dottuning.run_next_step()
    tuner.classifiers = Classifiers(
        singledot=MockClassifer('singledot'),
        doubledot=MockClassifer('doubledot'),
        dotregime=MockClassifer('doubledot')
    )

    sim_device.left_plunger.safety_voltage_range([-0.4, 0])
    sim_device.right_plunger.safety_voltage_range([-0.44, 0])
    meas_res = tuner.get_charge_diagram(
            sim_device,
            [sim_device.left_plunger, sim_device.right_plunger],
            use_safety_voltage_ranges=True,
            voltage_precision=0.005,
        )
    assert meas_res.status == sim_device.get_gate_status()
    assert meas_res.comment == f"Taking charge diagram of {[sim_device.left_plunger, sim_device.right_plunger]}."
    ds = nt.Dataset(meas_res.data_ids[-1], meas_res.db_name, db_folder=tmp_path)
    x_range = [ds.data.voltage_x.values[0], ds.data.voltage_x.values[-1]]
    y_range = [ds.data.voltage_y.values[0], ds.data.voltage_y.values[-1]]
    assert len(ds.data.voltage_x.values) == 80

    assert x_range == sim_device.left_plunger.safety_voltage_range()
    assert y_range == sim_device.right_plunger.safety_voltage_range()
    assert len(tuner.tuning_history.results[sim_device.name].tuningresults) == 1

    sim_device.current_valid_ranges(
    {
        sim_device.left_plunger.gate_id: [-0.3, 0],
        sim_device.right_plunger.gate_id: [-0.35, 0],
    })
    meas_res = tuner.get_charge_diagram(
            sim_device,
            [sim_device.left_plunger, sim_device.right_plunger],
            use_safety_voltage_ranges=False,
            voltage_precision=0.05,
        )
    ds = nt.Dataset(meas_res.data_ids[-1], meas_res.db_name, db_folder=tmp_path)
    x_range = [ds.data.voltage_x.values[0], ds.data.voltage_x.values[-1]]
    y_range = [ds.data.voltage_y.values[0], ds.data.voltage_y.values[-1]]

    assert x_range == sim_device.current_valid_ranges()[sim_device.left_plunger.gate_id]
    assert y_range == sim_device.current_valid_ranges()[sim_device.right_plunger.gate_id]


    with pytest.raises(ValueError):
        tuner.classifiers = Classifiers(pinchoff=MockClassifer('pinchoff'))
        _ = tuner.get_charge_diagram(
            sim_device,
            [sim_device.left_plunger, sim_device.right_plunger],
            use_safety_voltage_ranges=True,
        )
