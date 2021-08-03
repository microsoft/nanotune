import pytest

from nanotune.drivers.dac_interface import DACChannelInterface
from nanotune.drivers.mock_dac import MockDAC, MockDACChannel


def test_dacchannelinterface_methods():
    expected_methods = {
        "supports_hardware_ramp",
        "set_voltage",
        "get_voltage",
        "set_voltage_limit",
        "get_voltage_limit",
        "set_voltage_step",
        "get_voltage_step",
        "get_frequency",
        "set_frequency",
        "set_offset",
        "get_offset",
        "set_amplitude",
        "get_amplitude",
        "set_relay_state",
        "get_relay_state",
        "ramp_voltage",
        "set_ramp_rate",
        "get_ramp_rate",
        "get_waveform",
        "set_waveform",
    }

    f_set = DACChannelInterface.__abstractmethods__
    assert expected_methods == f_set


def test_dacinterface_attributes_and_methods():
    dac = MockDAC("dummy_dac", MockDACChannel)
    assert hasattr(dac, "channels")
    dac.close()
