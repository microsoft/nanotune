# pylint: disable=line-too-long, too-many-arguments, too-many-locals

""" Defines classes that allow a scripted simulation scenario to be defined."""

from abc import ABC, abstractmethod
from queue import SimpleQueue
from typing import Any, Tuple
import importlib
import logging
import ruamel.yaml

from sim.mock_pin import IMockPin
from sim.data_provider import IDataProvider
from sim.mock_device_registry import MockDeviceRegistry


class SimulationAction(ABC):

    """ Base class for all action types used with SimulationScenario """

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def run(self) -> None:
        """Performs the action"""
        raise NotImplementedError


class SetDataProviderAction(SimulationAction):

    """A simulation action that changes the data_provider an Mock Device Pin"""

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


class SimulationScenario(SimpleQueue[SimulationAction]):

    """Defines a simulation scenario, which is a sequence of actions to take when triggered to do so"""

    def __init__(self, name, actions=None):
        self._name = name
        if actions:
            for action in actions:
                self.put(action)

    @property
    def name(self):
        """ Name of the scenario"""
        return self._name

    def run_next_step(self) -> None:
        """Executes the next action.  If no actions remain, raises queue.Empty"""
        self.get_nowait().run()

    @classmethod
    def load_from_yaml(cls, yamlfile: str):

        """Creates a SimulationScenario and initializes it from the specified yaml file

        NOTE: Currently, the yaml file should list one data provider type per step.
              One SetDataProviderAction will be added to the scenario for each
              data provider defined in the yaml.
              ActionGroups are not yet supported in yaml.

              See tests/sim_scenario for a sample yaml file
        """

        def get_module_and_class(typename: str) -> Tuple[str, str]:
            module_name = ".".join(typename.split(".")[:-1])
            class_name = typename.split(".")[-1]
            return (module_name, class_name)

        def create_instance(full_type: str, **kwargs) -> Any:
            module_name, class_name = get_module_and_class(full_type)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(**kwargs)

        scenario = None
        with open(yamlfile) as file:

            data = ruamel.yaml.YAML().load(file)
            scenario_name = next(iter(data))
            scenario = SimulationScenario(scenario_name)

            for actions in data[scenario_name]:
                for name in actions:
                    action = actions[name]
                    sim_pin = MockDeviceRegistry.resolve_pin(action["sim_pin"])
                    provider = create_instance(
                        action["type"], **action["init"]
                    )

                    action = SetDataProviderAction(name, sim_pin, provider)
                    scenario.put(action)

        return scenario
