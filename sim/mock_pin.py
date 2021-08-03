from abc import ABC, abstractmethod

from sim.data_provider import IDataProvider


class IMockPin(ABC):
    """Interface for mock device pins"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the pin"""

    @abstractmethod
    def get_value(self) -> float:
        """return the pin's current value"""

    @abstractmethod
    def set_value(
        self,
        value: float,
    ) -> None:
        """Set the value on the pin.
        Raises an error if the data provider backing this pin is read only
        """

    @abstractmethod
    def set_data_provider(
        self,
        data_provider: IDataProvider,
    ) -> None:
        """change the data provider backing this pin"""

    @property
    @abstractmethod
    def settable(self) -> bool:
        """Indictates whether the value of this pin in settable or not"""
