import copy

import numpy as np
import pytest

import qcodes as qc
import nanotune as nt
from nanotune.device_tuner.tuner import (Tuner, set_back_valid_ranges,
                                         set_back_voltages)
from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.tests.mock_classifier import MockClassifer

atol = 1e-05


def test_set_back_voltages(gate_1, gate_2):
    gate_1.dc_voltage(-0.8)
    gate_2.dc_voltage(-0.9)
    assert gate_1.dc_voltage() == -0.8
    assert gate_2.dc_voltage() == -0.9

    with set_back_voltages([gate_1, gate_2]):
        assert gate_1.dc_voltage() == -0.8
        assert gate_2.dc_voltage() == -0.9
        gate_1.dc_voltage(-0.5)
        gate_2.dc_voltage(-0.4)
        assert gate_1.dc_voltage() == -0.5
        assert gate_2.dc_voltage() == -0.4

    assert gate_1.dc_voltage() == -0.8
    assert gate_2.dc_voltage() == -0.9


def test_set_back_valid_ranges(gate_1, gate_2):
    gate_1.current_valid_range([-0.8, -0.5])
    gate_2.current_valid_range([-0.9, -0.4])
    assert gate_1.current_valid_range() == [-0.8, -0.5]
    assert gate_2.current_valid_range() == [-0.9, -0.4]

    with set_back_valid_ranges([gate_1, gate_2]):
        assert gate_1.current_valid_range() == [-0.8, -0.5]
        assert gate_2.current_valid_range() == [-0.9, -0.4]
        gate_1.current_valid_range([-0.3, -0.4])
        gate_2.current_valid_range([-0.2, -0.1])
        assert gate_1.current_valid_range() == [-0.3, -0.4]
        assert gate_2.current_valid_range() == [-0.2, -0.1]

    assert gate_1.current_valid_range() == [-0.8, -0.5]
    assert gate_2.current_valid_range() == [-0.9, -0.4]


def test_tuner_init_and_attributes(tuner_default_input, tmp_path):
    tuner = Tuner(**tuner_default_input)
    data_settings = copy.deepcopy(tuner.data_settings())
    assert data_settings["db_name"] == "temp.db"
    assert data_settings["db_folder"] == str(tmp_path)
    assert data_settings["qc_experiment_id"] == 1

    new_data_settings = {"db_name": "other_temp.db"}
    tuner.data_settings(new_data_settings)
    data_settings.update(new_data_settings)
    assert tuner.data_settings() == data_settings

    assert tuner.setpoint_settings()["voltage_precision"] == 0.001

    tuner.close()


def test_update_normalization_constants(tuner_default_input, device_pinchoff, tmp_path):

    tuner = Tuner(**tuner_default_input)
    device_pinchoff.normalization_constants({})

    tuner.update_normalization_constants(device_pinchoff)
    updated_constants = device_pinchoff.normalization_constants()

    assert np.allclose(updated_constants["transport"], [0.0, 1.2], atol=atol)
    assert updated_constants["sensing"] != updated_constants["transport"]
    assert np.allclose(updated_constants["rf"], [0, 1], atol=atol)

    tuner.close()


def test_characterize_gates(tuner_default_input, device_pinchoff):
    tuner = Tuner(
        **tuner_default_input,
    )
    tuner.classifiers = {"pinchoff": MockClassifer("pinchoff")}
    result = tuner.characterize_gates(
        [device_pinchoff.left_barrier, device_pinchoff.left_barrier]
    )
    gate_name = "characterization_" + device_pinchoff.left_barrier.name
    assert gate_name in result.tuningresults.keys()
    print(result)
    tuningresult = result.tuningresults[gate_name]
    assert isinstance(tuningresult, TuningResult)
    assert tuningresult.success
    tuner.close()


def test_device_specific_settings(tuner_default_input, device_pinchoff):
    tuner = Tuner(
        **tuner_default_input,
    )
    original_setpoints = copy.deepcopy(tuner.setpoint_settings())
    original_classifiers = copy.deepcopy(tuner.classifiers)
    original_fit_options = copy.deepcopy(tuner.fit_options())

    assert "normalization_constants" not in tuner.data_settings().keys()
    n_csts = {"transport": (-0.3, 1.2), "sensing": (0.2, 0.8), "rf": (0, 1)}
    device_pinchoff.normalization_constants(n_csts)
    with tuner.device_specific_settings(device_pinchoff):
        assert tuner.data_settings()["normalization_constants"] == n_csts

        assert tuner.setpoint_settings() == original_setpoints
        assert tuner.classifiers == original_classifiers
        assert tuner.fit_options() == original_fit_options

    assert "normalization_constants" not in tuner.data_settings().keys()

    tuner.close()
