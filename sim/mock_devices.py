# pylint: disable=too-many-arguments, too-many-locals
from typing import List
from sim.data_providers import IDataProvider, StaticDataProvider
from sim.mock_device import IMockDevice
from sim.mock_device_registry import MockDeviceRegistry
from sim.mock_pin import IMockPin


class Pin(IMockPin):
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
class MockDevice(IMockDevice):
    """Base class for mock devices"""

    def __init__(self, name: str, pins: List[IMockPin], register: bool = True):
        self._name = name
        self._pins = {pin.name: pin for pin in pins}

        if register:
            MockDeviceRegistry.register(self)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "Pins  : {0}\n".format(
            ", ".join([str(pin) for pin in self._pins.values()])
        )

    def __getitem__(self, pin_name: str) -> IMockPin:
        return self._pins[pin_name]

    def get_pin(self, pin_name: str) -> IMockPin:
        return self._pins[pin_name]

    @property
    def name(self):
        return self._name


# pylint: enable=too-few-public-methods


class MockQuantumDot(MockDevice):
    """Represents a mock quantum dot device"""

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
    def src(self) -> IMockPin:
        """Source Pin"""

        return self["src"]

    @property
    def l_barrier(self) -> IMockPin:
        """Left Barrier Pin"""

        return self["l_barrier"]

    @property
    def l_plunger(self) -> IMockPin:
        """Left Plunger Pin"""

        return self["l_plunger"]

    @property
    def c_barrier(self) -> IMockPin:
        """Central Barrier Pin"""

        return self["c_barrier"]

    @property
    def r_plunger(self) -> IMockPin:
        """Right Plunger Pin"""

        return self["r_plunger"]

    @property
    def r_barrier(self) -> IMockPin:
        """Right Barrier Pin"""

        return self["r_barrier"]

    @property
    def drain(self) -> IMockPin:
        """Drain pin. This is the output device of the quantum dot."""

        return self["drain"]
