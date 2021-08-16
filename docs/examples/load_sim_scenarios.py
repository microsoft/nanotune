# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import nanotune as nt
from sim.simulation_scenario import SimulationScenario
from nanotune.device.device import NormalizationConstants

def sim_scenario_dottuning(station):
    nanotune_path = os.path.dirname(os.path.dirname(os.path.abspath(nt.__file__)))
    yamlfile = os.path.join(
        nanotune_path,
        "nanotune",
        "tests",
        "sim_scenarios",
        "sim_scenario_dottuning.yaml"
    )

    os.environ["db_path"] = nanotune_path
    scenario = SimulationScenario.load_from_yaml(yamlfile)
    station.sim_device.normalization_constants = NormalizationConstants(
        transport=(0, 1.93e-09), sensing=(0, 1), rf=(0, 1)
    )
    return scenario