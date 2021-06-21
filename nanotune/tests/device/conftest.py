import os
import pytest
import pathlib
from qcodes.tests.instrument_mocks import MockDAC as QcodesMockDAC
from nanotune.drivers.mock_dac import MockDAC, MockDACChannel
from nanotune.drivers.mock_readout_instruments import MockLockin, MockRF
from nanotune.device.device_channel import DeviceChannel

PARENT_DIR = pathlib.Path(__file__).parent.absolute()


@pytest.fixture(scope="session")
def qcodes_dac():
    dac = QcodesMockDAC('qcodes_dac', num_channels=3)
    try:
        yield dac
    finally:
        dac.close()



@pytest.fixture()
def chip_config():
    return os.path.join(PARENT_DIR, "chip.yml")


@pytest.fixture()
def device_on_chip(station, chip_config):
    if hasattr(station, "device_on_chip"):
        return station.device_on_chip

    station.load_config_file(chip_config)
    _chip = station.device_on_chip(station=station)
    return _chip

@pytest.fixture()
def gate(station):
    gate = DeviceChannel(
        station,
        station.dac.ch01,
        gate_id=0,
    )
    return gate


@pytest.fixture(name="moc_dac_server")
def _make_mock_dac_server():
    class DACClient:
        def __init__(self) -> None:
            self.socket = None

        def send(self, message: str) -> None:
            pass

    dac = DACClient()
    yield dac
