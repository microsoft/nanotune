import copy

import numpy as np
import pytest
import matplotlib.pyplot as plt
from dataclasses import asdict
from nanotune.device_tuner.tuner import Tuner, set_back_voltages
from nanotune.device.device import NormalizationConstants
from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.tuningstages.settings import Classifiers, DataSettings
from nanotune.tests.mock_classifier import MockClassifer

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


def test_tuner_init_and_attributes(tuner_default_input, tmp_path):
    tuner = Tuner(**tuner_default_input)
    assert tuner.data_settings.db_name == "temp.db"
    assert tuner.data_settings.db_folder == str(tmp_path)
    assert tuner.data_settings.experiment_id == None

    new_data_settings = DataSettings(db_name="other_temp.db")
    data_settings = DataSettings(**asdict(tuner.data_settings))

    tuner.data_settings = new_data_settings
    data_settings.update(new_data_settings)
    assert tuner.data_settings == data_settings

    assert tuner.setpoint_settings.voltage_precision == 0.001

    tuner.close()


def test_update_normalization_constants(
    tuner_default_input,
    sim_device_pinchoff,
):
    tuner = Tuner(**tuner_default_input)
    previous_constants = sim_device_pinchoff.normalization_constants
    sim_device_pinchoff.normalization_constants = NormalizationConstants()

    tuner.update_normalization_constants(sim_device_pinchoff)
    updated_constants = sim_device_pinchoff.normalization_constants

    assert updated_constants.transport == previous_constants.transport
    assert updated_constants.sensing == (0., 1.)
    assert updated_constants.rf == (0., 1.)

    tuner.close()


def test_characterize_gates(
    tuner_default_input,
    sim_device_pinchoff,
    pinchoff_classifier,
):
    tuner = Tuner(**tuner_default_input)
    tuner.classifiers = Classifiers(pinchoff=pinchoff_classifier)

    measurement_result = tuner.characterize_gates(
        sim_device_pinchoff,
        [sim_device_pinchoff.right_plunger],
        use_safety_voltage_ranges=True,
    )
    plt.close('all')

    stage_name = "characterization_" + sim_device_pinchoff.right_plunger.name
    tuningresult = measurement_result.tuningresults[stage_name]
    assert isinstance(tuningresult, TuningResult)
    assert tuningresult.status == sim_device_pinchoff.get_gate_status()
    comment = f"Characterizing {[sim_device_pinchoff.right_plunger]}."
    assert tuningresult.comment == comment


    sim_device_pinchoff.current_valid_ranges(
        {sim_device_pinchoff.right_plunger.gate_id: (-0.2, 0)}
    )
    measurement_result = tuner.characterize_gates(
        sim_device_pinchoff,
        [sim_device_pinchoff.right_plunger],
        use_safety_voltage_ranges=False,
        comment = "first run",
        iterate=False,
    )
    tuningresult = measurement_result.tuningresults[stage_name]
    assert tuningresult.comment == "first run"
    assert not tuningresult.success

    measurement_result = tuner.characterize_gates(
        sim_device_pinchoff,
        [sim_device_pinchoff.right_plunger],
        use_safety_voltage_ranges=False,
        iterate=True,
    )
    tuningresult = measurement_result.tuningresults[stage_name]
    assert tuningresult.success
    assert len(tuningresult.data_ids) == 2

    with pytest.raises(KeyError):
        tuner.classifiers = Classifiers()
        _ = tuner.characterize_gates(
            sim_device_pinchoff,
            [sim_device_pinchoff.right_plunger],
            use_safety_voltage_ranges=True,
        )

    tuner.close()


def test_measurement_setpoint_settings(
    tuner_default_input,
    sim_device_pinchoff,
):
    tuner = Tuner(**tuner_default_input)

    tuner.close()

