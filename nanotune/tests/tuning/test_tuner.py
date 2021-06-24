import copy

import numpy as np
import pytest
from dataclasses import asdict
from nanotune.device_tuner.tuner import Tuner, set_back_voltages
from nanotune.device.device import NormalizationConstants
from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.tuningstages.settings import DataSettings
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


def test_update_normalization_constants(tuner_default_input, device, tmp_path):

    tuner = Tuner(**tuner_default_input)
    device.normalization_constants = {}

    tuner.update_normalization_constants(device)
    updated_constants = device.normalization_constants

    assert np.allclose(updated_constants["transport"], [0.0, 1.2], atol=atol)
    assert updated_constants["sensing"] != updated_constants["transport"]
    assert np.allclose(updated_constants["rf"], [0, 1], atol=atol)

    tuner.close()


def test_characterize_gates(tuner_default_input, device):
    tuner = Tuner(
        **tuner_default_input,
    )
    tuner.classifiers = {"pinchoff": MockClassifer("pinchoff")}
    result = tuner.characterize_gates(
        [device.top_barrier, device.top_barrier]
    )
    gate_name = "characterization_" + device.top_barrier.name
    assert gate_name in result.tuningresults.keys()
    print(result)
    tuningresult = result.tuningresults[gate_name]
    assert isinstance(tuningresult, TuningResult)
    assert tuningresult.success
    tuner.close()


def test_measurement_setpoint_settings(tuner_default_input, device):
    tuner = Tuner(
        **tuner_default_input,
    )

    tuner.close()
