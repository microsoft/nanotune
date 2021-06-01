import logging
import os
from queue import Empty

import sim
from sim.data_providers import StaticDataProvider
from sim.mock_devices import MockQuantumDot, Pin
from sim.simulation_scenario import (ActionGroup, SetDataProviderAction,
                                     SimulationScenario)


def test_load_from_yaml():

    """Tests initializing a SimulationScenario from yaml, and using it to drive a mock device"""

    simroot = os.path.dirname(os.path.abspath(sim.__file__))
    yamlfile = os.path.join(simroot, "test", "test_simulation_scenario.yaml")
    nt_root = os.path.dirname(os.path.dirname(os.path.abspath(sim.__file__)))
    os.environ["sim_db_path"] = nt_root

    qdsim = MockQuantumDot("qdsim")
    scenario = SimulationScenario.load_from_yaml(yamlfile)

    # Test Simple Action
    scenario.run_next_step()
    assert qdsim.drain.get_value() == 10.0

    # Test ActionGroup
    scenario.run_next_step()
    assert qdsim.l_plunger.get_value() == -1.0
    assert qdsim.r_plunger.get_value() == 1.0

    # Test QcodesDataProvider
    scenario.run_next_step()

    qdsim.l_barrier.set_value(-0.1)
    readout1 = qdsim.drain.get_value()

    qdsim.l_barrier.set_value(-0.2)
    readout2 = qdsim.drain.get_value()
    assert readout1 != readout2

def test_typical_simulation_scenario():

    """Tests a typical scenario containing ActionGroups and leaf actions"""

    o1 = Pin("O1")
    o2 = Pin("O2")

    scenario = SimulationScenario(
        "typical scenario",
        [
            ActionGroup(
                "Initialize All Data Providers",
                [
                    SetDataProviderAction(
                        "Set o1 to 1.0", o1, StaticDataProvider(1.0)
                    ),
                    SetDataProviderAction(
                        "Set o2 to 2.0", o2, StaticDataProvider(2.0)
                    ),
                ],
            ),
            SetDataProviderAction(
                "Set o1 to 3.0", o1, StaticDataProvider(3.0)
            ),
            SetDataProviderAction(
                "Set o2 to 4.0", o2, StaticDataProvider(4.0)
            ),
            ActionGroup(
                "Set all inputs to zero",
                [
                    SetDataProviderAction(
                        "Set o1 to 0.0", o1, StaticDataProvider(0.0)
                    ),
                    SetDataProviderAction(
                        "Set o2 to 0.0", o2, StaticDataProvider(0.0)
                    ),
                ],
            ),
        ],
    )

    # Defines the expected outputs after each action is executed
    expected_outputs = [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (0, 0, 0.0)]

    try:
        i_expected = iter(expected_outputs)
        n_actions = scenario.action_count
        n_actions_performed = 0

        while True:
            scenario.run_next_step()

            # Validate we havent gotten here more times than expected
            n_actions_performed = n_actions_performed + 1
            assert n_actions_performed <= n_actions

            # Validate the actions resulted in the output pins having the expected values
            expected_output_pin_values = next(i_expected)
            assert o1.get_value() == expected_output_pin_values[0]
            assert o2.get_value() == expected_output_pin_values[1]

    except Empty:
        # validate the correct number of actions ran, and the output pins are set to values
        # specified by the last action
        assert n_actions_performed == n_actions
        assert o1.get_value() == expected_outputs[-1][0]
        assert o2.get_value() == expected_outputs[-1][1]

def test_empty_simulation_scenario():
    """Tests attempting to run steps on an empty scenario"""
    scenario = SimulationScenario("empty scenario")

    try:
        scenario.run_next_step()
        # Should not get here
        assert False
    except Empty:
        pass

def test_dynamic_simulation_scenario():

    """Tests dynamically adding actions to a scenario and executing them"""

    def test_next_step(
        scenario, pin, expected_pre_value, expected_post_value
    ):
        assert pin.get_value() == expected_pre_value
        scenario.run_next_step()
        assert pin.get_value() == expected_post_value

    """ Tests running a scenario where steps are dynamically added """
    scenario = SimulationScenario("dynamic scenario")

    o1 = Pin("o1")

    # Test dynamically adding and running a single action
    scenario.append(
        SetDataProviderAction(
            "Set o1 to 1.0", o1, StaticDataProvider(1.0)
        )
    )
    test_next_step(scenario, o1, 0.0, 1.0)

    # Test dynamically adding multiple actions
    scenario.append(
        SetDataProviderAction(
            "Set o1 to 2.0", o1, StaticDataProvider(2.0)
        )
    )
    scenario.append(
        SetDataProviderAction(
            "Set o1 to 3.0", o1, StaticDataProvider(3.0)
        )
    )

    test_next_step(scenario, o1, 1.0, 2.0)
    test_next_step(scenario, o1, 2.0, 3.0)

    # Test trying to run with the scenario now being empty
    assert scenario.empty
    try:
        scenario.run_next_step()
        assert False  # should not get here
    except Empty:
        pass

    # Re-validate dynamically adding a scenario after trying to run it while it was empty
    scenario.append(
        SetDataProviderAction(
            "Set o1 to 4.0", o1, StaticDataProvider(4.0)
        )
    )
    test_next_step(scenario, o1, 3.0, 4.0)
