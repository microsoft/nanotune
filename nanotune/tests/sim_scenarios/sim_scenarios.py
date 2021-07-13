# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import pytest

from sim.simulation_scenario import SimulationScenario
from nanotune.device.device import NormalizationConstants


@pytest.fixture(scope="function")
def sim_scenario_init_ranges(nanotune_path, sim_station):
    yamlfile = os.path.join(
        nanotune_path,
        "nanotune",
        "tests",
        "sim_scenarios",
        "sim_scenario_init_ranges.yaml"
    )

    os.environ["db_path"] = nanotune_path
    scenario = SimulationScenario.load_from_yaml(yamlfile)
    sim_station.sim_device.normalization_constants = NormalizationConstants(
        transport=(0, 2.45e-09), sensing=(0, 1), rf=(0, 1)
    )

    return scenario


@pytest.fixture(scope="function")
def sim_scenario_device_characterization(nanotune_path, sim_station):
    yamlfile = os.path.join(
        nanotune_path,
        "nanotune",
        "tests",
        "sim_scenarios",
        "sim_scenario_device_characterization.yaml"
    )

    os.environ["db_path"] = nanotune_path
    scenario = SimulationScenario.load_from_yaml(yamlfile)
    sim_station.sim_device.normalization_constants = NormalizationConstants(
        transport=(0, 2.45e-09), sensing=(0, 1), rf=(0, 1)
    )

    return scenario


@pytest.fixture(scope="function")
def sim_scenario_dottuning(nanotune_path, sim_station):
    yamlfile = os.path.join(
        nanotune_path,
        "nanotune",
        "tests",
        "sim_scenarios",
        "sim_scenario_dottuning.yaml"
    )

    os.environ["db_path"] = nanotune_path
    scenario = SimulationScenario.load_from_yaml(yamlfile)
    sim_station.sim_device.normalization_constants = NormalizationConstants(
        transport=(0, 1.93e-09), sensing=(0, 1), rf=(0, 1)
    )

    return scenario
