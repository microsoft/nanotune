from abc import ABC, abstractmethod

from sim.mock_pin import IMockPin

class IMockDevice(ABC):

    """ Base interface for all mock devices """

    @property
    @abstractmethod
    def name(self) -> str:
        """ Mock device name """

    @abstractmethod
    def get_pin(self, pin_name: str) -> IMockPin:
        """ Get pin by name """
