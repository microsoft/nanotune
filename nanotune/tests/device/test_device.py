from dataclasses import asdict
import pytest

import nanotune as nt
from nanotune.drivers.dac_interface import RelayState
from nanotune.device.device import (Readout,
    _add_station_and_label_to_channel_init, NormalizationConstants,
    ReadoutMethods)


def test_device_init_defaults(station):
    device = nt.Device(
        "test_doubledot",
        station,
    )

    assert not device.gates
    assert not device.ohmics
    assert not device.readout
    assert not device.initial_valid_ranges()
    assert device.quality() == 0
    assert device.normalization_constants == NormalizationConstants()
    assert device.current_valid_ranges() == device.initial_valid_ranges()
    assert not device.transition_voltages()

    assert device.snapshot()


def test_device_init(device, station):

    assert len(device.gates) == 2
    assert len(device.ohmics) == 1

    assert device.gates[0] == device.top_barrier
    assert device.ohmics[0] == device.left_ohmic

    assert device.current_valid_ranges() == {0: [-0.5, 0], 1: [-3, 0]}
    assert device.initial_valid_ranges() == {0: [-3, 0], 1: [-3, 0]}
    assert device.transition_voltages() == {0: -0.4, 1: None}

    norm_constants = device.normalization_constants
    assert norm_constants.transport == (0, 2)
    assert norm_constants.sensing == (-0.3, 0.6)

    assert isinstance(device.readout, Readout)

    with pytest.raises(KeyError):
        _ = nt.Device(
            "test_doubledot",
            station,
            readout={'some_readout': None}
        )


def test_device_normalization_constants_setter(device):

    device.normalization_constants = {
        'transport': (-1.9, -1.2), 'rf': (-2, 0)
    }
    assert device.normalization_constants.transport == (-1.9, -1.2)
    assert device.normalization_constants.rf == (-2, 0)

    norm_dict = asdict(device.normalization_constants)
    assert device.metadata['normalization_constants'] == norm_dict

    device.normalization_constants = NormalizationConstants(
        transport=(-1, 1.1)
    )
    assert device.normalization_constants.transport == (-1, 1.1)
    assert device.normalization_constants.sensing == (0, 1)

    norm_dict = asdict(device.normalization_constants)
    assert device.metadata['normalization_constants'] == norm_dict


def test_device_main_readout_method_setter(device):
    device.main_readout_method = ReadoutMethods.transport
    assert device.main_readout_method == ReadoutMethods.transport

    with pytest.raises(ValueError):
        device.main_readout_method = 'transport'

    with pytest.raises(ValueError):
        device.main_readout_method = ReadoutMethods.rf


def test_device_initial_valid_ranges(device):
    ranges = {0: [-0.99, -0.1], 1: [-0.2, -0.13]}
    device.initial_valid_ranges(ranges)
    assert device._initial_valid_ranges == ranges
    assert device.initial_valid_ranges() == ranges


def test_device_current_valid_ranges(device):
    ranges = {0: [-0.99, -0.1], 1: [-0.2, -0.13]}
    device.current_valid_ranges(ranges)
    assert device._current_valid_ranges == ranges
    assert device.current_valid_ranges() == ranges


def test_voltage_range_setter(device):
    ranges = {0: [-0.99, -0.1], 1: [-0.2, -0.13]}
    new_sub_dict = {'top_barrier': [-0.6, -0.3], 1: [-0.1, 0]}
    new_ranges = device.voltage_range_setter(ranges, new_sub_dict)
    assert new_ranges == {0: [-0.6, -0.3], 1: [-0.1, 0]}

    ranges = {0: [-0.98, -0.11], 1: [-0.21, -0.14]}
    new_sub_dict = {device.gates[0]: [-0.7, -0.5], 1: [-0.2, 0]}
    new_ranges = device.voltage_range_setter(ranges, new_sub_dict)
    assert new_ranges == {0: [-0.7, -0.5], 1: [-0.2, 0]}


    device.gates[0].safety_voltage_range([-1, 0])
    ranges = {0: [-0.5, -0.11], 1: [-0.21, -0.14]}
    new_sub_dict = {device.gates[0]: [-1.1, 0.5], 1: [-0.2, 0]}
    new_ranges = device.voltage_range_setter(ranges, new_sub_dict)
    assert new_ranges == {0: [-1, 0], 1: [-0.2, 0]}


def test_device_get_gate_id(device, gate_1):
    assert device.get_gate_id(0) == 0
    assert device.get_gate_id(device.left_plunger) == 1
    assert device.get_gate_id('top_barrier') == 0

    with pytest.raises(ValueError):
        device.get_gate_id(3)
    with pytest.raises(ValueError):
        device.get_gate_id('center')
    with pytest.raises(ValueError):
        device.get_gate_id(gate_1)
    with pytest.raises(ValueError):
        device.get_gate_id([0])


def test_device_check_and_update_new_voltage_range(device):
    v_range = [-1, 0]
    safety_voltage_range = [-2, 0]
    new_range = device.check_and_update_new_voltage_range(v_range, safety_voltage_range)
    assert new_range == v_range

    new_range = device.check_and_update_new_voltage_range([0, -1], safety_voltage_range)
    assert new_range == v_range

    v_range = [-2.3, 0.5]
    safety_voltage_range = [-2, 0]
    new_range = device.check_and_update_new_voltage_range(v_range, safety_voltage_range)
    assert new_range == [-2, 0]

    v_range = [-3, -2.5]
    safety_voltage_range = [-2, 0]
    with pytest.raises(ValueError):
        device.check_and_update_new_voltage_range(v_range, safety_voltage_range)

    v_range = [1, 2]
    safety_voltage_range = [-2, 0]
    with pytest.raises(ValueError):
        device.check_and_update_new_voltage_range(v_range, safety_voltage_range)

def test_device_methods(device):

    for gate in device.gates:
        gate.voltage(-1)
        gate.relay_state(RelayState.ground)

    device.all_gates_to_highest()
    device.ground_gates()
    for gate in device.gates:
        assert gate.voltage() == 0
        assert gate.relay_state() == RelayState.ground

    device.top_barrier.safety_voltage_range([-3, 0])
    device.all_gates_to_lowest()
    for gate in device.gates:
        assert gate.voltage() == -3

    device.ohmics[0].relay_state(RelayState.bus)
    device.float_ohmics()
    assert device.ohmics[0].relay_state() == RelayState.floating


def test_device_transition_voltages_setter(device):
    t_vs = {0: -0.7, 1: -0.5}
    device.transition_voltages(t_vs)
    trans_voltages = device.transition_voltages()
    assert trans_voltages == t_vs

    t_vs = {'top_barrier': -0.6, device.left_plunger: -0.8}
    device.transition_voltages(t_vs)
    trans_voltages = device.transition_voltages()
    assert trans_voltages == {0: -0.6, 1: -0.8}

    device.current_valid_ranges({0: [0, 0]})
    device.top_barrier.voltage(0)
    device.top_barrier.safety_voltage_range([-1, 0])
    device.transition_voltages({0: -1.6})
    trans_voltages = device.transition_voltages()
    assert trans_voltages == {0: -1, 1: -0.8}

    device.transition_voltages({0: 1.6})
    trans_voltages = device.transition_voltages()
    assert trans_voltages == {0: 0, 1: -0.8}


def test_device_rename_gate_identifier(device):
    mapping_to_rename = {'top_barrier': [1, 0, 2], device.left_plunger: -0.1}
    new_mapping = device.rename_gate_identifier(mapping_to_rename)
    assert new_mapping != mapping_to_rename
    assert new_mapping == {0: [1, 0, 2], 1: -0.1}


def test_device_initialize_channel_lists(station):
    device = nt.Device(
        "test_doubledot",
        station,
    )
    channels_input_mapping = {
        'type': 'nanotune.device.device_channel.DeviceChannel',
        'top_barrier': {
            'channel': 'dac.ch01', 'gate_id': 0,
            'station': station,
            },
        'left_plunger': {
            'channel': 'dac.ch02', 'gate_id': 1,
            'station': station,
        },
        'other_gate': {
            'channel': 'dac.ch02',
            'station': station,
        },
        'left_ohmic': {
            'channel': 'dac.ch03', 'ohmic_id': 0,
            'station': station,
        },
        'right_ohmic': {
            'channel': 'dac.ch04',
            'station': station,
    }

    }
    device._create_and_add_channels(
                station=station,
                channels=channels_input_mapping,
            )
    (gates,
     ohmics,
     gate_labels,
     gates_list,
     ohmics_list) = device.initialize_channel_lists(
        channels_input_mapping)
    ## gates and ohmic without IDs are ignored
    assert len(gates) == 2
    assert len(ohmics) == 1
    assert len(gates_list) == 2
    assert len(ohmics_list) == 1
    assert len(gate_labels) == 2
    assert gates[0] == device.top_barrier
    assert gates[1] == device.left_plunger
    assert ohmics[0] == device.left_ohmic
    assert gate_labels == {0: 'top_barrier', 1: 'left_plunger'}


def test_get_gate_status(device):
    device.top_barrier.safety_voltage_range([-3, 0])
    device.left_plunger.safety_voltage_range([-3, 0])

    device.top_barrier.voltage(-1.23)
    device.left_plunger.voltage(-0.98)
    device.current_valid_ranges({0: [-2, -1.5], 1: [-1, 0]})

    status = device.get_gate_status()
    assert status['top_barrier'] == {
        'current_valid_range': [-2, -1.5], 'voltage': -1.23}
    assert status['left_plunger'] == {
        'current_valid_range': [-1, 0], 'voltage': -0.98}


def test_device_fill_missing_voltage_ranges(device):
    v_ranges_dict = {0: [-1, 0]}
    new_dict = device._fill_missing_voltage_ranges(v_ranges_dict)
    assert new_dict == {0: [-1, 0], 1: [-3, 0]}

    v_ranges_dict = {0: [-0.5, 0], 1: [-1.9, 0]}
    new_dict = device._fill_missing_voltage_ranges(v_ranges_dict)
    assert new_dict == v_ranges_dict


def test_add_station_and_label_to_channel_init(station):
    channels_input_mapping = {
        'type': 'nanotune.device.device_channel.DeviceChannel',
        'top_barrier': {
            'channel': 'dac.ch01', 'gate_id': 0,
            },
        'other_gate': {
            'channel': 'dac.ch02',
        },
        'left_ohmic': {
            'channel': 'dac.ch03', 'ohmic_id': 0,
        },
        'right_ohmic': {
            'channel': 'dac.ch04',
    }}
    channels_input_mapping = _add_station_and_label_to_channel_init(
        station,
        channels_input_mapping)

    correct_mapping = {
    'type': 'nanotune.device.device_channel.DeviceChannel',
    'top_barrier': {
        'channel': 'dac.ch01', 'gate_id': 0,
        'station': station,
        'label': 'top_barrier',
        },
    'other_gate': {
        'channel': 'dac.ch02',
        'station': station,
        'label': 'other_gate',
    },
    'left_ohmic': {
        'channel': 'dac.ch03', 'ohmic_id': 0,
        'station': station,
        'label': 'left_ohmic',
    },
    'right_ohmic': {
        'channel': 'dac.ch04',
        'station': station,
        'label': 'right_ohmic',
    }}
    assert channels_input_mapping == correct_mapping
