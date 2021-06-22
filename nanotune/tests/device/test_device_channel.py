# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import pytest
import time
from nanotune.device.device_channel import DeviceChannel
from nanotune.drivers.dac_interface import RelayState


def test_device_channel_init(station):
    gate_1 = DeviceChannel(
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
    assert gate_1.name == station.dac.ch01.name
    assert gate_1._channel == station.dac.ch01

    assert not gate_1.supports_hardware_ramp

    assert gate_1.label == 'TB'
    assert gate_1.safety_voltage_range() == [-2.2, 0]
    assert gate_1._channel.voltage.vals.valid_values == (-2.2, 0)
    assert gate_1.voltage.vals.valid_values == (-2.2, 0)

    assert gate_1.inter_delay == 0.04
    assert gate_1._channel.voltage.inter_delay == 0.04
    assert gate_1.post_delay == 0.034
    assert gate_1._channel.voltage.post_delay == 0.034

    assert gate_1.max_voltage_step == 0.03
    assert gate_1._channel.voltage.step == 0.03
    assert not gate_1.use_ramp()
    assert gate_1.ramp_rate() == 0.7

    assert gate_1.gate_id == 0

    assert gate_1.metadata['label'] == 'TB'
    assert gate_1.metadata['gate_id'] == 0
    assert gate_1.metadata['ohmic_id'] is None
    assert gate_1.metadata['test_meta'] == 'duh'

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

def test_device_channel_parameters(station, gate_1):

    station.dac.ch01.voltage(-0.434)
    assert gate_1.voltage() == -0.434
    gate_1.max_voltage_step = 2
    gate_1.voltage(-1.1)
    assert station.dac.ch01.voltage() == -1.1

    gate_1.safety_voltage_range([-2, 0])
    assert gate_1.safety_voltage_range() == [-2, 0]

    gate_1.voltage(-1.5)
    with pytest.raises(ValueError):
        gate_1.safety_voltage_range([-1, 0])

    gate_1.ramp_rate(0.2)
    assert gate_1.ramp_rate() == 0.2

    gate_1.use_ramp(True)
    assert gate_1.use_ramp()

    gate_1.relay_state(RelayState.smc)
    assert gate_1.relay_state() == RelayState.smc

    gate_1.amplitude(0.03)
    assert gate_1.amplitude() == 0.03

    with pytest.raises(ValueError):
        gate_1.offset(0.02)
    gate_1.offset(-0.02)
    assert gate_1.offset() == -0.02

    gate_1.frequency(83)
    assert gate_1.frequency() == 83

def test_device_channel_attributes(gate_1):

    with pytest.raises(AttributeError):
        gate_1.gate_id = 10

    with pytest.raises(AttributeError):
        gate_1.ohmic_id = 10

    with pytest.raises(AttributeError):
        gate_1.supports_hardware_ramp = True

    gate_1.post_delay = 0.15
    assert gate_1._channel.voltage.post_delay == 0.15
    gate_1.inter_delay = 0.17
    assert gate_1._channel.voltage.inter_delay == 0.17
    gate_1.max_voltage_step = 0.19
    assert gate_1._channel.voltage.step == 0.19


def test_device_channel_relay_settings(station):
    gate_1 = DeviceChannel(
        station,
        station.dac.ch01,
        gate_id=0,
    )
    gate_1.ground()
    assert gate_1.relay_state() == RelayState.ground

    gate_1.float_relay()
    assert gate_1.relay_state() == RelayState.floating

def test_device_channel_voltage_setter(gate_1):
    gate_1.voltage(0)
    gate_1.max_voltage_step = 2
    gate_1.safety_voltage_range([-1, 0])
    with pytest.raises(ValueError):
        gate_1.voltage(-2)
    with pytest.raises(ValueError):
        gate_1.voltage(2)

    gate_1.post_delay = 0
    gate_1.inter_delay = 0.5
    gate_1.max_voltage_step = 0.5
    gate_1.voltage(0)
    gate_1.use_ramp(True)
    start = time.time()
    gate_1.voltage(-1)
    end_time = time.time()
    assert end_time - start > 1

    gate_1.use_ramp(False)
    gate_1.max_voltage_step = 2
    gate_1.voltage(0)
    start = time.time()
    gate_1.voltage(-1)
    end_time = time.time()
    assert end_time - start < 0.1

    assert gate_1.inter_delay == 0.5
    assert gate_1.max_voltage_step == 2

    gate_1.max_voltage_step = 0.05
    with pytest.raises(ValueError):
        gate_1.voltage(0)


def test_set_temp_inter_delay_and_step(gate_1):
    gate_1.inter_delay = 0.5
    gate_1.max_voltage_step = 0.06
    with gate_1._set_temp_inter_delay_and_step(0.012, 0.03):
        assert gate_1.inter_delay == 0.012
        assert gate_1.max_voltage_step == 0.03

    assert gate_1.inter_delay == 0.5
    assert gate_1.max_voltage_step == 0.06



