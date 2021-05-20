# pylint: disable=too-many-arguments, too-many-locals
from typing import List
from sim.data_providers import IDataProvider, StaticDataProvider
from sim.simulator import ISimulator
from sim.simulator_registry import SimulatorRegistry
from sim.pin import IPin


class Pin(IPin):
    """Default Pin Device.  Uses an IDataProvider to represent the
    value on the pin device
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._data_provider: IDataProvider = StaticDataProvider(0.0)

    def __repr__(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        """Name of the pin"""

        return self._name

    def get_value(self) -> float:
        """Gets the current value on the input pin.  Compatible with qcodes
        Parameter get_cmd argument.
        """
        return self._data_provider.value

    def set_value(
        self,
        value: float,
    ) -> None:
        """Set the value on the pin.
        Raises an error if the data provider backing this pin is read only
        """
        self._data_provider.set_value(value)

    def set_data_provider(
        self,
        data_provider: IDataProvider,
    ) -> None:
        """change the data provider backing this pin"""

        self._data_provider = data_provider

    @property
    def settable(self) -> bool:
        """Indictates whether the value of this pin in settable or not"""
        return self._data_provider.settable


# pylint: disable=too-few-public-methods
class SimulatedDevice(ISimulator):
    """Base class for simulated devices"""

    def __init__(self, name: str, pins: List[IPin], register: bool = True):
        self._name = name
        self._pins = {pin.name: pin for pin in pins}

        if register:
            SimulatorRegistry.register(self)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "Pins  : {0}\n".format(
            ", ".join([str(pin) for pin in self._pins.values()])
        )

    def __getitem__(self, pin_name: str) -> IPin:
        return self._pins[pin_name]

    def get_pin(self, pin_name: str) -> IPin:
        return self._pins[pin_name]

    @property
    def name(self):
        return self._name


# pylint: enable=too-few-public-methods


class QuantumDotSim(SimulatedDevice):
    """Represents all gates on a quantum dot."""

    def __init__(self, name: str):
        super().__init__(
            name,
            [
                Pin("src"),
                Pin("l_barrier"),
                Pin("l_plunger"),
                Pin("c_barrier"),
                Pin("r_plunger"),
                Pin("r_barrier"),
                Pin("drain"),
            ],
        )

    @property
    def src(self) -> IPin:
        """Source Pin"""

        return self["src"]

    @property
    def l_barrier(self) -> IPin:
        """Left Barrier Pin"""

        return self["l_barrier"]

    @property
    def l_plunger(self) -> IPin:
        """Left Plunger Pin"""

        return self["l_plunger"]

    @property
    def c_barrier(self) -> IPin:
        """Central Barrier Pin"""

        return self["c_barrier"]

    @property
    def r_plunger(self) -> IPin:
        """Right Plunger Pin"""

        return self["r_plunger"]

    @property
    def r_barrier(self) -> IPin:
        """Right Barrier Pin"""

        return self["r_barrier"]

    @property
    def drain(self) -> IPin:
        """Drain pin. This is the output device of the quantum dot."""

        return self["drain"]
