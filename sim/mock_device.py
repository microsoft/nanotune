from abc import ABC, abstractmethod, abstractproperty
from sim.mock_pin import IMockPin

class IMockDevice(ABC):

    """ Base interface for all mock devices """

    @abstractproperty
    def name(self):
        """ Mock device name """
        raise NotImplementedError

    @abstractmethod
    def get_pin(self, pin_name: str) -> IMockPin:
        """ Get pin by name """
        raise NotImplementedError
