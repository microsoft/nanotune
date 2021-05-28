import pytest

from nanotune.device.ohmic import Ohmic


def test_ohmic_attributes_after_init(dummy_device, dummy_dac):
    ohmic = Ohmic(
        dummy_device,
        dummy_dac,
        channel_id=1,
        ohmic_id=1,
        name="test_ohmic",
        label="test_label",
    )

    assert ohmic.channel_id() == 1
    assert ohmic.ohmic_id() == 1
    assert ohmic.name == dummy_device.name + "_" + "test_ohmic"
    assert ohmic.label() == "test_label"


def test_ohmic_get_snapshot(dummy_device, dummy_dac):
    ch_id = 1
    ohmic = Ohmic(
        dummy_device,
        dummy_dac,
        channel_id=ch_id,
        ohmic_id=1,
        name="test_ohmic",
        label="test_label",
    )

    snap = ohmic.snapshot_base()
    assert snap["dac_channel"] == dummy_dac.nt_channels[ch_id].name
