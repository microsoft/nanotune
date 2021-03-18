import pytest



@pytest.fixture(name="moc_dac_server")
def _make_mock_dac_server():
    class DACClient:
        def __init__(self) -> None:
            self.socket = None

        def send(self, message: str) -> None:
            pass

    dac = DACClient()
    try:
        yield dac
    finally:
        pass
        # dac.close()
