import numpy as np
import pytest
from qcodes import Parameter

from nanotune.device.gate import Gate
from nanotune.drivers.dac_interface import DACChannelInterface, DACInterface


def test_gate_attributes_after_init(dummy_device, dummy_dac):
    safety_range = (-2, 0)
    delay = 0.01
    max_jump = 0.01
    ramp_rate = 0.05
    use_ramp = False
    gate = Gate(
        dummy_device,
        dummy_dac,
        channel_id=1,
        layout_id=1,
        name="test_gate",
        label="test_label",
        safety_range=safety_range,
        delay=delay,
        ramp_rate=ramp_rate,
        use_ramp=use_ramp,
        max_jump=max_jump,
    )

    # inter_delay will be overwritten if/when ramp is set.
    assert gate.channel_id() == 1
    assert gate.layout_id() == 1
    assert gate.name == dummy_device.name + "_" + "test_gate"
    assert gate.label() == "test_label"
    assert gate.inter_delay() == delay
    assert gate.post_delay() == delay
    assert gate.max_jump() == max_jump
    assert gate.step() == max_jump
    assert gate.use_ramp() == use_ramp
    assert gate.ramp_rate() == ramp_rate
    assert gate.current_valid_range() == list(safety_range)
    assert gate.transition_voltage() == 0
    assert gate.amplitude() == 0
    assert gate.frequency() == 0
    assert gate.offset() == 0

    with pytest.raises(AttributeError):
        gate.delay()

    assert np.allclose(gate.safety_range(), safety_range, rtol=0.1)

    assert not gate.has_ramp

    assert isinstance(gate.dc_voltage, Parameter)
    assert isinstance(gate._dac_channel, DACChannelInterface)


def test_gate_ramp(dummy_device, dummy_dac):
    # test if delay and step parameter as set correctly when ramping
    # Assume DAC does not support hardware ramping of voltages and that
    # a software ramp, implemented in qcodes, is used
    safety_range = (-2, 0)
    delay = 0.01
    max_jump = 0.01
    ramp_rate = 0.05
    use_ramp = True
    gate = Gate(
        dummy_device,
        dummy_dac,
        channel_id=1,
        layout_id=1,
        name="test_gate",
        label="test_label",
        safety_range=safety_range,
        delay=delay,
        ramp_rate=ramp_rate,
        max_jump=max_jump,
    )

    gate.has_ramp = False
    gate.use_ramp(use_ramp)
    calculated_inter_delay = max_jump / ramp_rate

    gate.dc_voltage(-0.5)

    assert gate.inter_delay() == calculated_inter_delay
    assert gate.post_delay() == delay
    assert gate.max_jump() == max_jump
    assert gate.step() == max_jump


def test_gate_get_snapshot(dummy_device, dummy_dac):
    ch_id = 1
    gate = Gate(dummy_device, dummy_dac, channel_id=ch_id, layout_id=1)

    snap = gate.snapshot_base()
    assert snap["dac_channel"] == dummy_dac.nt_channels[ch_id].name


def test_gate_attributes_setters(dummy_device, dummy_dac):
    gate = Gate(dummy_device, dummy_dac, channel_id=1, layout_id=1)

    delay = 0.03
    gate.inter_delay(delay)
    assert gate.inter_delay() == delay
    gate.post_delay(delay)
    assert gate.post_delay() == delay

    max_jump = 0.03
    gate.max_jump(max_jump)
    assert gate.max_jump() == max_jump
    gate.step(max_jump)
    assert gate.step() == max_jump

    use_ramp = False
    ramp_rate = 0.05
    gate.use_ramp(use_ramp)
    assert gate.use_ramp() == use_ramp
    gate.ramp_rate(ramp_rate)
    assert gate.ramp_rate() == ramp_rate

    # current_valid_range setter should check against safety ranges
    safety_range = [-2, 0]
    gate.safety_range(safety_range)
    gate.current_valid_range([-4, 0])
    assert gate.current_valid_range() == safety_range
    gate.current_valid_range([-1, 0])
    assert gate.current_valid_range() == [-1, 0]
    gate.current_valid_range([-1, 1])
    assert gate.current_valid_range() == [-1, 0]

    too_negative = safety_range[0] - 1
    too_positive = safety_range[1] + 1
    just_right = safety_range[0] + 0.5

    with pytest.raises(ValueError):
        gate.transition_voltage(too_negative)
    with pytest.raises(ValueError):
        gate.transition_voltage(too_positive)
    gate.transition_voltage(just_right)
    assert gate.transition_voltage() == just_right

    with pytest.raises(ValueError):
        gate.dc_voltage(too_negative)
    with pytest.raises(ValueError):
        gate.dc_voltage(too_positive)
    gate.dc_voltage(just_right)
    assert gate.dc_voltage() == just_right

    with pytest.raises(ValueError):
        gate.offset(too_negative)
    with pytest.raises(ValueError):
        gate.offset(too_positive)
    gate.offset(just_right)
    assert gate.offset() == just_right

    gate.amplitude(0.1)
    assert gate.amplitude() == 0.1
    gate.frequency(83)
    assert gate.frequency() == 83
