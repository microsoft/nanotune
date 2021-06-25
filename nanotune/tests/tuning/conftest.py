from nanotune.tests.mock_classifier import MockClassifer
import pytest
from nanotune.tuningstages.settings import (DataSettings, SetpointSettings,
    Classifiers)

@pytest.fixture(scope="function")
def tuner_default_input(tmp_path):
    settings = {
        "name": "test_tuner",
        "data_settings": DataSettings(
            db_name="temp.db",
            db_folder=str(tmp_path),
        ),
        "classifiers": Classifiers(pinchoff=MockClassifer(category="pinchoff")),
        "setpoint_settings": SetpointSettings(
            voltage_precision=0.001,
            ranges_to_sweep=[(-1, 0)],
            safety_voltage_ranges=[(-3, 0)],
        ),
    }
    yield settings
