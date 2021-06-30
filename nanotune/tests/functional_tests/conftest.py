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

    # qd_mock_instrument = MockDoubleQuantumDotInstrument('qd_mock_instrument')
    scenario = SimulationScenario.load_from_yaml(yamlfile)
    # sim_station.add_component(qd_mock_instrument)
    return scenario


@pytest.fixture(scope="function")
def sim_device_init_ranges(sim_scenario_init_ranges, sim_station, sim_device):

    qd_mock_instrument = sim_station.qd_mock_instrument
    sim_device.central_barrier.voltage = qd_mock_instrument.central_barrier
    sim_device.left_barrier.voltage = qd_mock_instrument.left_barrier
    sim_device.right_barrier.voltage = qd_mock_instrument.right_barrier
    sim_device.top_barrier.voltage = qd_mock_instrument.top_barrier

    sim_device.normalization_constants = {
        'transport': (0, 2.45e-09), 'sensing': (0, 1), 'rf': (0, 1)
    }
    return sim_device