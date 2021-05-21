from abc import ABC, abstractmethod, abstractproperty
from sim.data_provider import IDataProvider


class IMockPin(ABC):
    """Interface for mock device pins"""

    @abstractproperty
    def name(self):
        """Name of the pin"""
        raise NotImplementedError

    @abstractmethod
    def get_value(self) -> float:
        """return the pin's current value"""
        raise NotImplementedError

    @abstractmethod
    def set_value(
        self,
        value: float,
    ) -> None:
        """Set the value on the pin.
        Raises an error if the data provider backing this pin is read only
        """
        raise NotImplementedError

    @abstractmethod
    def set_data_provider(
        self,
        data_provider: IDataProvider,
    ) -> None:
        """change the data provider backing this pin"""
        raise NotImplementedError

    @abstractproperty
    def settable(self) -> bool:
        """Indictates whether the value of this pin in settable or not"""
        raise NotImplementedError
