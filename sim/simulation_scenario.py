# pylint: disable=line-too-long, too-many-arguments, too-many-locals

""" Defines classes that allow a scripted simulation scenario to be defined."""

import importlib
import logging
from abc import ABC, abstractclassmethod, abstractmethod
from queue import SimpleQueue
from typing import Any, Mapping, Tuple

import ruamel.yaml

from sim.data_providers import IDataProvider
from sim.mock_device_registry import MockDeviceRegistry
from sim.mock_pin import IMockPin
from sim.serializable import ISerializable


class SimulationAction(ISerializable):

    """ Base class for all action types used with SimulationScenario """

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def run(self) -> None:
        """Performs the action"""


class SetDataProviderAction(SimulationAction):

    """A simulation action that changes the data_provider an Mock Device Pin"""

    @classmethod
    def make(cls, **kwargs):

        """ ISerializable override to create an instance of this class """

        name = kwargs["name"]
        pin = MockDeviceRegistry.resolve_pin(kwargs["pin"])
        data_provider = ISerializable.create_from_type(**kwargs["data_provider"])
        return cls(name, pin, data_provider)

    def __init__(self, name: str, pin: IMockPin, data_provider: IDataProvider):
        super().__init__(name)

        if not data_provider:
            raise ValueError("A valid data provider must be specified")

        if not pin:
            raise ValueError(
                "A valid data provider container must be specified"
            )

        if not hasattr(pin, "set_data_provider"):
            raise RuntimeError(
                "The specified pin is not a valid data_provider container"
            )

        self._data_provider = data_provider
        self._data_provider_container = pin

    def run(self) -> None:
        logging.info("Running %s : %s", self.__class__.__name__, self._name)
        self._data_provider_container.set_data_provider(self._data_provider)


class ActionGroup(SimulationAction):

    """Represents a group of actions that will be performed when this action is run.
    For example, if you wanted to change the data provider on multiple output pins at once"""

    @classmethod
    def make(cls, **kwargs):

        """ ISerializable override to create an instance of this class """

        name = kwargs["name"]
        actions_node = kwargs["actions"]
        actions = [ISerializable.create_from_named_node(action)
                   for action in actions_node]
        return cls(name, actions)

    def __init__(self, name, actions):
        super().__init__(name)
        self._actions = list(actions)

    def add_action(self, action: SimulationAction) -> None:

        """Adds an action to this step that will set the specified data provider onto the specified container"""
        self._actions.append(action)

    def run(self) -> None:
        logging.info("Running %s : %s", self.__class__.__name__, self._name)
        for action in self._actions:
            action.run()


class SimulationScenario():

    """Defines a simulation scenario, which is a sequence of actions to take when triggered to do so"""

    def __init__(self, name, actions=None):
        self._name = name
        self._actions : "SimpleQueue[SimulationAction]" = SimpleQueue()
        if actions:
            for action in actions:
                self._actions.put(action)

    @property
    def name(self):
        """ Name of the scenario"""
        return self._name

    @property
    def action_count(self) -> int:
        """ The number of actions remaining in the scenario """
        return self._actions.qsize()

    @property
    def empty(self) -> bool:
        """ True if the scenario has no remaining actions """
        return bool(self._actions.empty)

    def append(self, action :SimulationAction) -> None:
        """Enqueues an action to the scenario"""
        self._actions.put(action)

    def run_next_step(self) -> None:
        """Executes the next action.  If no actions remain, raises queue.Empty"""
        self._actions.get_nowait().run()

    @classmethod
    def load_from_yaml(cls, yamlfile: str):

        """Creates a SimulationScenario and initializes it from the specified yaml file

        NOTE: Currently, the yaml file should list one data provider type per step.
              One SetDataProviderAction will be added to the scenario for each
              data provider defined in the yaml.
              ActionGroups are not yet supported in yaml.

              See tests/sim_scenario for a sample yaml file
        """

        scenario = None
        with open(yamlfile) as file:

            data = ruamel.yaml.YAML().load(file)
            scenario_name = next(iter(data))
            scenario = SimulationScenario(scenario_name)

            for action_node in data[scenario_name]:
                action = ISerializable.create_from_named_node(action_node)
                scenario.append(action)

        return scenario
