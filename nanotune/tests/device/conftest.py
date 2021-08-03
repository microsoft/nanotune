import pytest
from qcodes.tests.instrument_mocks import MockDAC as QcodesMockDAC
from qcodes.instrument.delegate.delegate_instrument import DelegateInstrument


@pytest.fixture(scope="function")
def qcodes_dac():
    dac = QcodesMockDAC('qcodes_dac', num_channels=3)
    try:
        yield dac
    finally:
        dac.close()


@pytest.fixture(scope="function")
def delegate_instrument(station):
    instr = DelegateInstrument(
        'dummy',
        station,
        parameters={'test_param': 'lockin.phase'}
    )
    return instr


@pytest.fixture(name="moc_dac_server")
def _make_mock_dac_server():
    class DACClient:
        def __init__(self) -> None:
            self.socket = None

        def send(self, message: str) -> None:
            pass

    dac = DACClient()
    yield dac
