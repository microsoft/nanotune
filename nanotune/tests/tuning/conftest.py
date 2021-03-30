import pytest


@pytest.fixture(scope="function")
def tuner_default_input(tmp_path):
    settings = {
        'name': 'test_tuner',
        'data_settings': {'db_name': 'temp.db',
                          'db_folder': str(tmp_path)},
        'classifiers': {},
        'setpoint_settings': {'voltage_precision': 0.001},
    }
    yield settings
