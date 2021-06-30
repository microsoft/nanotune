import os
import pytest
import nanotune as nt
from sim.data_providers import QcodesDataProvider


@pytest.fixture(scope="function")
def sim_device_gatecharacterization2d(
    tuning_data_path,
    sim_station,
    sim_device):
    db_path = os.path.join(tuning_data_path, "gatecharacterization2d.db")
    qd_mock_instrument = sim_station.qd_mock_instrument

    data_provider = QcodesDataProvider(
        [qd_mock_instrument.mock_device.left_plunger,
            qd_mock_instrument.mock_device.right_plunger],
        db_path, 'nanotune_fivedot_general', 167)
    # Configure the simulator's drain pin to use the backing data
    qd_mock_instrument.mock_device.drain.set_data_provider(
        data_provider)

    sim_device.left_plunger.voltage = qd_mock_instrument.left_plunger
    sim_device.right_plunger.voltage = qd_mock_instrument.right_plunger
    sim_device.all_gates_to_highest()

    sim_device.normalization_constants = {
        'transport': [0, 1.1e-09], 'rf': [0, 1], 'sensing': [0, 1]}
    return sim_device