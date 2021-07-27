# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import dataclasses
import pytest
import nanotune as nt
from nanotune.tuningstages.settings import (SetpointSettings, DataSettings,
    Classifiers)
from nanotune.device.device import NormalizationConstants


def test_data_settings_attributes(tmp_path):

    assert sorted(DataSettings.__dataclass_fields__.keys()) == sorted(
        ['db_folder',
        'db_name',
        'dot_signal_threshold',
        'experiment_id',
        'noise_floor',
        'normalization_constants',
        'segment_db_folder',
        'segment_db_name',
        'segment_experiment_id',
        'segment_size']
    )
    settings = DataSettings()
    assert isinstance(settings.db_name, str)
    assert isinstance(settings.db_folder, str)
    assert settings.normalization_constants == NormalizationConstants()
    assert settings.experiment_id is None
    assert settings.segment_db_name == f'segmented_{settings.db_name}'
    assert settings.segment_db_folder == settings.db_folder
    assert settings.segment_experiment_id is None
    assert settings.segment_size == 0.05

    settings.update({'normalization_constants':
        {'transport': (0.1, 1.1), 'sensing': (0, 1), 'rf': (0, 1 )}
        })
    assert settings.normalization_constants == NormalizationConstants(
        transport=(0.1, 1.1))


def test_setpoint_settings_attributes():
    assert sorted(SetpointSettings.__dataclass_fields__.keys()) == sorted([
        'voltage_precision', 'parameters_to_sweep', 'safety_voltage_ranges',
        'ranges_to_sweep', 'setpoint_method', 'high_res_precisions'])


def test_data_settings_update():
    data_settings = DataSettings(db_name='temp.db')
    data_settings.update(
        {'normalization_constants': NormalizationConstants(
            transport=(0.1, 0.1)),
        'experiment_id': 1,

        }
    )
    assert data_settings.normalization_constants == NormalizationConstants(
        transport=(0.1, 0.1))


def test_setpoint_settings_update():
    setpoint_settings = SetpointSettings(voltage_precision=0.3)
    setpoint_settings.update(
        {'voltage_precision': 0.2,
        'safety_voltage_ranges': [(-2, 0)],
        }
    )
    assert setpoint_settings.voltage_precision == 0.2
    assert setpoint_settings.safety_voltage_ranges == [(-2, 0)]
    assert not setpoint_settings.ranges_to_sweep
    assert not setpoint_settings.parameters_to_sweep
    assert not setpoint_settings.setpoint_method

    setpoint_settings.update(SetpointSettings(voltage_precision=0.01))
    assert setpoint_settings.voltage_precision == 0.01
    assert not setpoint_settings.safety_voltage_ranges
    assert not setpoint_settings.ranges_to_sweep
    assert not setpoint_settings.parameters_to_sweep
    assert not setpoint_settings.setpoint_method

    with pytest.raises(KeyError):
        setpoint_settings.update({'params_to_sweep': 'dac.ch02'})


def test_classifiers_attributes():
    assert sorted(Classifiers.__dataclass_fields__.keys()) == sorted([
        'pinchoff', 'singledot', 'doubledot', 'dotregime'])
