import pytest
from nanotune.tuningstages.settings import (DataSettings, SetpointSettings,
    Classifiers)

@pytest.fixture(scope="function")
def tuner_default_input(tmp_path, gate_1):
    settings = {
        "name": "test_tuner",
        "data_settings": DataSettings(
            db_name="temp.db",
            db_folder=str(tmp_path),
        ),
        "classifiers": Classifiers(),
        "setpoint_settings": SetpointSettings(
            voltage_precision=0.001,
            ranges_to_sweep=[(-0.3, 0)],
            safety_voltage_ranges=[(-3, 0)],
        ),
    }
    yield settings
