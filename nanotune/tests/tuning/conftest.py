import pytest


@pytest.fixture(scope="function")
def tuner_default_input(tmp_path, gate_1):
    settings = {
        'name': 'test_tuner',
        'data_settings': {
            'db_name': 'temp.db',
            'db_folder': str(tmp_path),
        },
        'classifiers': {},
        'setpoint_settings': {
            'voltage_precision': 0.001,
            'current_valid_ranges': [(-0.3, 0)],
            'safety_voltage_ranges': [(-3, 0)],
        },
    }
    yield settings
