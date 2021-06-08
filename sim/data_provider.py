# pylint: disable=too-many-arguments, too-many-locals

from abc import ABC, abstractmethod
from typing import Union

import xarray as xr
from sim.serializable import ISerializable


class IDataProvider(ISerializable):
    """Interface for all data providers.
    A DataProvider serves as the backing data for simulation
    """

    @property
    @abstractmethod
    def settable(self) -> bool:
        """Indicates whether this data provider allows its value to be set
        be set by calling set_value
        """

    @abstractmethod
    def get_value(self) -> float:
        """Returns the current value of this data provider"""

    @abstractmethod
    def set_value(self, value: float) -> None:
        """Set the value of this data provider.
        Raises exception if this data provider is read-only
        """

    @property
    @abstractmethod
    def raw_data(self) -> Union[xr.DataArray, xr.Dataset]:
        """Returns the raw data backing this provider as an
        xarray.DataArray
        """


class DataProvider(IDataProvider):
    """Base class for data providers"""

    def __init__(self, settable: bool):
        self._settable = settable

    @property
    def settable(self) -> bool:
        """Indicates whether this data provider allows its value to
        be set by calling set_value
        """
        return self._settable
