# pylint: disable=too-many-arguments, too-many-locals
from typing import Mapping, Sequence

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
        return self._data_provider.get_value()

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

    def __init__(self, name: str, pins: Sequence[IMockPin], register: bool = True):
        self._name = name
        self._pins : Mapping[str, IMockPin] = {pin.name: pin for pin in pins}

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


class MockSingleQuantumDot(MockDevice):
    """ Mock device for a 3 gate single quantum dot """
    def __init__(self, name: str):
        super().__init__(
            name,
            [
                Pin("src"),
                Pin("drain"),
                Pin("left_barrier"),
                Pin("right_barrier"),
                Pin("plunger"),
            ]
        )

    @property
    def src(self) -> IMockPin:
        """ Source Pin """
        return self["src"]

    @property
    def drain(self) -> IMockPin:
        """ Drain Pin """
        return self["drain"]

    @property
    def left_barrier(self) -> IMockPin:
        """ Left Barrier Gate Pin """
        return self["left_barrier"]

    @property
    def right_barrier(self) -> IMockPin:
        """ Right Barrier Gate Pin """
        return self["right_barrier"]

    @property
    def plunger(self) -> IMockPin:
        """ Central Plunger Gate Pin """
        return self["plunger"]



class MockDoubleQuantumDot(MockDevice):
    """Represents a mock quantum dot device"""

    def __init__(self, name: str):
        super().__init__(
            name,
            [
                Pin("src"),
                Pin("left_barrier"),
                Pin("left_plunger"),
                Pin("central_barrier"),
                Pin("right_plunger"),
                Pin("right_barrier"),
                Pin("drain"),
            ],
        )

    @property
    def src(self) -> IMockPin:
        """Source Pin"""

        return self["src"]

    @property
    def left_barrier(self) -> IMockPin:
        """Left Barrier Pin"""

        return self["left_barrier"]

    @property
    def left_plunger(self) -> IMockPin:
        """Left Plunger Pin"""

        return self["left_plunger"]

    @property
    def central_barrier(self) -> IMockPin:
        """Central Barrier Pin"""

        return self["central_barrier"]

    @property
    def right_plunger(self) -> IMockPin:
        """Right Plunger Pin"""

        return self["right_plunger"]

    @property
    def right_barrier(self) -> IMockPin:
        """Right Barrier Pin"""

        return self["right_barrier"]

    @property
    def drain(self) -> IMockPin:
        """Drain pin. This is the output device of the quantum dot."""

        return self["drain"]
