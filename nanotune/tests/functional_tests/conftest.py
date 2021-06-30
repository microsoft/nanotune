# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import pytest

from sim.simulation_scenario import SimulationScenario


@pytest.fixture(scope="function")
def sim_scenario_init_ranges(nanotune_path, sim_station):
    yamlfile = os.path.join(
        nanotune_path,
        "nanotune",
        "tests",
        "functional_tests",
        "sim_scenario_init_ranges.yaml"
    )

    os.environ["db_path"] = nanotune_path
    scenario = SimulationScenario.load_from_yaml(yamlfile)
    return scenario


@pytest.fixture(scope="function")
def sim_device_playback(sim_station, sim_device):

    qd_mock_instrument = sim_station.qd_mock_instrument
    sim_device.top_barrier.voltage = qd_mock_instrument.top_barrier
    sim_device.left_barrier.voltage = qd_mock_instrument.left_barrier
    sim_device.left_plunger.voltage = qd_mock_instrument.left_plunger
    sim_device.central_barrier.voltage = qd_mock_instrument.central_barrier
    sim_device.right_plunger.voltage = qd_mock_instrument.right_plunger
    sim_device.right_barrier.voltage = qd_mock_instrument.right_barrier

    sim_device.normalization_constants = {
        'transport': (0, 2.45e-09), 'sensing': (0, 1), 'rf': (0, 1)
    }
    return sim_device


@pytest.fixture(scope="function")
def sim_scenario_device_characterization(nanotune_path, sim_station):
    yamlfile = os.path.join(
        nanotune_path,
        "nanotune",
        "tests",
        "functional_tests",
        "sim_scenario_device_characterization.yaml"
    )

    os.environ["db_path"] = nanotune_path
    scenario = SimulationScenario.load_from_yaml(yamlfile)
    return scenario
