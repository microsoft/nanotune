""" Defines MockPin """

# pylint: disable=too-many-arguments, too-many-locals
from sim.data_provider import IDataProvider
from sim.data_providers import StaticDataProvider
from sim.mock_pin import IMockPin


class MockPin(IMockPin):
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
