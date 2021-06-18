""" Defines MockDevice, which is the base class for all mock devices """

# pylint: disable=too-many-arguments, too-many-locals
from typing import Mapping, Sequence

from sim.mock_device import IMockDevice
from sim.mock_device_registry import MockDeviceRegistry
from sim.mock_pin import IMockPin


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
