from typing import Dict, Optional
from sim.simulator import ISimulator
from sim.pin import IPin


class SimulatorRegistry:

    """Acts as a singelton registry for simulator instances
       so they can be easily accessed by name from other modules
    """

    _sim_registry: Dict[str, ISimulator] = {}

    @classmethod
    def register(
        cls, simulator: ISimulator, name: Optional[str] = None
    ) -> None:

        """Register a simulator using the simulator's name, or using the optionally provided name"""

        simname = simulator.name if name is None else name
        cls._sim_registry[simname] = simulator

    @classmethod
    def get(cls, name: str) -> ISimulator:

        """Retrieve the simulator registered under the given name"""

        return cls._sim_registry[name]

    @classmethod
    def resolve_pin(cls, full_pin_name: str) -> IPin:

        """Returns the pin of a simulator given full pin name
        e.g.  "qdsim.drain" would be the name of the "drain" pin on the "qdsim" simulator"""

        sim_name, pin_name = full_pin_name.split(".")
        return cls.get(sim_name).get_pin(pin_name)
