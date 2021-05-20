from abc import ABC, abstractmethod, abstractproperty
from sim.pin import IPin

class ISimulator(ABC):

    """ Base interface for all simulators """

    @abstractproperty
    def name(self):
        """ Simulator Name """
        raise NotImplementedError

    @abstractmethod
    def get_pin(self, pin_name: str) -> IPin:
        """ Get pin by name """
        raise NotImplementedError
