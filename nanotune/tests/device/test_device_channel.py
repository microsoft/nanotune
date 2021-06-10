# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import pytest
import time
import numpy as np
from nanotune.device.device_channel import DeviceChannel


def test_device_channel_init(station):
    gate = DeviceChannel(
        station.dac,
        'top_barrier',
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

    assert not gate.has_ramp

    assert gate.label == 'TB'
    assert gate.safety_voltage_range() == [-2.2, 0]
    assert gate.inter_delay == 0.04
    assert gate._channel.get_voltage_inter_delay() == 0.04
    assert gate.post_delay == 0.034
    assert gate._channel.get_voltage_post_delay() == 0.034

    assert gate.max_voltage_step() == 0.03
    assert gate._channel.get_voltage_step() == 0.03
    assert not gate.use_ramp()
    assert gate.ramp_rate() == 0.7

    assert gate.gate_id == 0

    assert gate.metadata['label'] == label
    assert gate.metadata['gate_id'] == 0
    assert gate.metadata['ohmic_id'] is None
    assert gate.metadata['test_meta'] == 'duh'

    gate = DeviceChannel(
        station.dac,
        'top_barrier',
        'dac.ch02',
    )
    assert gate._channel == station.dac.ch02

def test_device_channel_init_exceptions(station, qcodes_dac):
    with pytest.raises(ValueError):
        _ = DeviceChannel(
            station.dac,
            'top_barrier',
            station.dac.ch01,
            gate_id=0,
            ohmic_id=0,
        )

    with pytest.raises(ValueError):
        _ = DeviceChannel(
            station.dac,
            'top_barrier',
            station.dac.ch01.voltage,
        )

    with pytest.raises(AssertionError):
        _ = DeviceChannel(
            qcodes_dac,
            'top_barrier',
            qcodes_dac.ch01,
        )

def test_device_channel_parameters(station, gate):

    station.dac.ch01.voltage(-0.434)
    assert gate.voltage() == -0.434
    gate.voltage(-1.1)
    assert station.dac.ch01.voltage() == -1.1

    gate.safety_voltage_range([-2, 0])
    assert gate.safety_voltage_range() == [-2, 0]

    gate.max_voltage_step(0.1)
    assert gate.max_voltage_step() == 0.1

    gate.ramp_rate(0.2)
    assert gate.max_voltage_step() == 0.2

    gate.use_ramp(True)
    assert gate.max_voltage_step()

    gate.relay_state('smc')
    assert gate.relay_state() == 'smc'

    gate.amplitude(0.03)
    assert gate.amplitude() == 0.03

    gate.offset(0.02)
    assert gate.amplitude() == 0.02

    gate.frequency(83)
    assert gate.frequency() == 83

def test_device_channel_attributes(gate):

    with pytest.raises(AttributeError):
        gate.gate_id = 10

    with pytest.raises(AttributeError):
        gate.ohmic_id = 10

    with pytest.raises(AttributeError):
        gate.has_ramp = True

    gate.post_delay = 0.15
    assert gate.voltage.post_delay == 0.15
    gate.inter_delay = 0.17
    assert gate.voltage.inter_delay == 0.17
    gate.step = 0.19
    assert gate.voltage.step == 0.19

def test_device_channel_relay_settings(gate):
    gate = DeviceChannel(
        station.dac,
        'top_barrier',
        station.dac.ch01,
        gate_id=0,
    )
    gate.ground()
    assert gate.relay_state() == 'ground'

    gate.float_relay()
    assert gate.relay_state() == 'float'

def test_device_channel_voltage_setter(gate):

    gate.safety_voltage_range([-1, 0])
    with pytest.raises(ValueError):
        gate.voltage(-2)
    with pytest.raises(ValueError):
        gate.voltage(2)

    gate.ramp_rate(0.4)
    gate.post_delay(0)
    gate.use_ramp(True)
    gate.voltage(0)

    start = time.time()
    gate.voltage(-1)
    end_time = time.time()
    assert np.isclose((end_time - start), 2.5, atol=0.1)

    gate.use_ramp(False)
    gate.voltage(0)
    start = time.time()
    gate.voltage(-1)
    end_time = time.time()
    assert np.isclose((end_time - start), 0, atol=0.1)


