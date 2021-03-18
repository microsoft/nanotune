import pytest

from nanotune.drivers.dac_interface import DACChannelInterface, DACInterface
from nanotune.tests.dac_mocks import DummyDACChannel, DummyDAC


def test_dacchannelinterface_methods():
    expected_methods = {
        "set_relay_state",
        "get_post_delay",
        "set_offset",
        "get_frequency",
        "set_filter",
        "set_step",
        "set_frequency",
        "get_waveform",
        "ramp_voltage",
        "supports_hardware_ramp",
        "set_waveform",
        "get_limit_rate",
        "set_dc_voltage",
        "get_offset",
        "set_dc_voltage_limit",
        "get_filter",
        "set_label",
        "set_ramp_rate",
        "get_relay_state",
        "set_amplitude",
        "get_step",
        "get_dc_voltage",
        "get_label",
        "set_post_delay",
        "get_ramp_rate",
        "get_amplitude",
        "set_inter_delay",
        "get_inter_delay",
        "set_limit_rate",
    }

    f_set = DACChannelInterface.__abstractmethods__
    assert expected_methods == f_set


def test_dacinterface_attributes_and_methods():
    dac = DummyDAC("dummy_dac", DummyDACChannel)
    assert hasattr(dac, "nt_channels")
    dac.close()
