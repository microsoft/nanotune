import pytest
import qcodes as qc
import nanotune as nt


def test_device_init_defaults():
    device = nt.Device('test_fivedot', 'fivedot')

    assert not device.gates
    assert not device.ohmics

    assert device.snapshot()
    device.close()

def test_device_gates(device_fivedot_inputs):
    device = nt.Device(name='test_fivedot',
        device_type='fivedot',
        **device_fivedot_inputs,
        )
    assert len(device.gates) == 5
    assert len(device.ohmics) == 1
    assert len(device.sensor_gates) == 0

    init_valid_ranges = {}
    for gate in device.gates:
        init_valid_ranges[gate.layout_id()] = gate.safety_range()
    assert device.initial_valid_ranges() == init_valid_ranges

    assert device.quality() == 0
    readout_methods = device.readout_methods()
    assert all(key in readout_methods for key in ['dc_current', 'dc_sensor'])
    assert isinstance(readout_methods['dc_current'], qc.Parameter)

    n_sct = device.normalization_constants()
    assert isinstance(n_sct, dict)
    assert all(key in n_sct for key in ['dc_current', 'dc_sensor', 'rf'])

    assert device.sensor_side() == "left"
    with pytest.raises(ValueError):
        device.sensor_side("anywhere")
    device.close()

def test_device_gate_methods(device_fivedot_inputs):
    device = nt.Device(name='test_fivedot',
        device_type='fivedot',
        **device_fivedot_inputs,
        )

    for gate in device.gates:
        gate.dc_voltage(-1)
        gate.relay_state('unknown')
    device.all_gates_to_highest()
    device.ground_gates()
    for gate in device.gates:
        assert gate.dc_voltage() == 0
        assert gate.relay_state() == 'ground'

    device.ohmics[0].relay_state('unknown')
    device.float_ohmics()
    assert device.ohmics[0].relay_state() == 'float'
    device.close()

def test_device_snapshot(device_fivedot_inputs):
    device = nt.Device(name='test_fivedot',
        device_type='fivedot',
        **device_fivedot_inputs,
        )
    snap = device.snapshot()
    assert isinstance(snap['parameters']['readout_methods'], dict)
    snap_keys = list(snap['parameters']['readout_methods'].keys())
    readout_methods = device.readout_methods().keys()
    assert all(key in readout_methods for key in snap_keys)
    for key in snap_keys:
        assert isinstance(snap['parameters']['readout_methods'][key], dict)

    device.close()


