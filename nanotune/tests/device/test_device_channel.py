# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import pytest
import time
from nanotune.device.device_channel import DeviceChannel
from nanotune.drivers.dac_interface import RelayState


def test_device_channel_init(station):
    gate = DeviceChannel(
        station,
        station.dac.ch01,
        gate_id=0,
        label='TB',
        safety_voltage_range=(-2.2, 0),
        inter_delay=0.04,
        post_delay=0.034,
        max_voltage_step=0.03,
        use_ramp=False,
        ramp_rate=0.7,
        metadata={'test_meta': 'duh'},
    )
    assert gate.name == station.dac.ch01.name
    assert gate._channel == station.dac.ch01

    assert not gate.supports_hardware_ramp

    assert gate.label == 'TB'
    assert gate.safety_voltage_range() == [-2.2, 0]
    assert gate._channel.voltage.vals.valid_values == (-2.2, 0)
    assert gate.voltage.vals.valid_values == (-2.2, 0)

    assert gate.inter_delay == 0.04
    assert gate._channel.voltage.inter_delay == 0.04
    assert gate.post_delay == 0.034
    assert gate._channel.voltage.post_delay == 0.034

    assert gate.max_voltage_step == 0.03
    assert gate._channel.voltage.step == 0.03
    assert not gate.use_ramp()
    assert gate.ramp_rate() == 0.7

    assert gate.gate_id == 0

    assert gate.metadata['label'] == 'TB'
    assert gate.metadata['gate_id'] == 0
    assert gate.metadata['ohmic_id'] is None
    assert gate.metadata['test_meta'] == 'duh'

    gate = DeviceChannel(
        station,
        'dac.ch02',
    )
    assert gate._channel == station.dac.ch02

def test_device_channel_init_exceptions(station, qcodes_dac):
    with pytest.raises(ValueError):
        _ = DeviceChannel(
            station,
            station.dac.ch01,
            gate_id=0,
            ohmic_id=0,
        )

    with pytest.raises(ValueError):
        _ = DeviceChannel(
            station,
            station.dac.ch01.voltage,
        )

    with pytest.raises(AssertionError):
        _ = DeviceChannel(
            station,
            qcodes_dac.ch01,
        )

def test_device_channel_parameters(station, gate):

    station.dac.ch01.voltage(-0.434)
    assert gate.voltage() == -0.434
    gate.max_voltage_step = 2
    gate.voltage(-1.1)
    assert station.dac.ch01.voltage() == -1.1

    gate.safety_voltage_range([-2, 0])
    assert gate.safety_voltage_range() == [-2, 0]

    gate.voltage(-1.5)
    with pytest.raises(ValueError):
        gate.safety_voltage_range([-1, 0])

    gate.ramp_rate(0.2)
    assert gate.ramp_rate() == 0.2

    gate.use_ramp(True)
    assert gate.use_ramp()

    gate.relay_state(RelayState.smc)
    assert gate.relay_state() == RelayState.smc

    gate.amplitude(0.03)
    assert gate.amplitude() == 0.03

    with pytest.raises(ValueError):
        gate.offset(0.02)
    gate.offset(-0.02)
    assert gate.offset() == -0.02

    gate.frequency(83)
    assert gate.frequency() == 83

def test_device_channel_attributes(gate):

    with pytest.raises(AttributeError):
        gate.gate_id = 10

    with pytest.raises(AttributeError):
        gate.ohmic_id = 10

    with pytest.raises(AttributeError):
        gate.supports_hardware_ramp = True

    gate.post_delay = 0.15
    assert gate._channel.voltage.post_delay == 0.15
    gate.inter_delay = 0.17
    assert gate._channel.voltage.inter_delay == 0.17
    gate.max_voltage_step = 0.19
    assert gate._channel.voltage.step == 0.19


def test_device_channel_relay_settings(gate, station):
    gate = DeviceChannel(
        station,
        station.dac.ch01,
        gate_id=0,
    )
    gate.ground()
    assert gate.relay_state() == RelayState.ground

    gate.float_relay()
    assert gate.relay_state() == RelayState.floating

def test_device_channel_voltage_setter(gate):
    gate.voltage(0)
    gate.max_voltage_step = 2
    gate.safety_voltage_range([-1, 0])
    with pytest.raises(ValueError):
        gate.voltage(-2)
    with pytest.raises(ValueError):
        gate.voltage(2)

    gate.post_delay = 0
    gate.inter_delay = 0.5
    gate.max_voltage_step = 0.5
    gate.voltage(0)
    gate.use_ramp(True)
    start = time.time()
    gate.voltage(-1)
    end_time = time.time()
    assert end_time - start > 1

    gate.use_ramp(False)
    gate.max_voltage_step = 2
    gate.voltage(0)
    start = time.time()
    gate.voltage(-1)
    end_time = time.time()
    assert end_time - start < 0.1

    assert gate.inter_delay == 0.5
    assert gate.max_voltage_step == 2

    gate.max_voltage_step = 0.05
    with pytest.raises(ValueError):
        gate.voltage(0)


def test_set_temp_inter_delay_and_step(gate):
    gate.inter_delay = 0.5
    gate.max_voltage_step = 0.06
    with gate._set_temp_inter_delay_and_step(0.012, 0.03):
        assert gate.inter_delay == 0.012
        assert gate.max_voltage_step == 0.03

    assert gate.inter_delay == 0.5
    assert gate.max_voltage_step == 0.06



