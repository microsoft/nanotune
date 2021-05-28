# pylint: disable=too-many-arguments, too-many-locals

from abc import ABC, abstractmethod
from typing import Union

import xarray as xr


class IDataProvider(ABC):
    """Interface for all data providers.
    A DataProvider serves as the backing data for simulation
    """

    @property
    @abstractmethod
    def settable(self) -> bool:
        """Indicates whether this data provider allows its value to be set
        be set by calling set_value
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(self) -> float:
        """Returns the current value of this data provider"""
        raise NotImplementedError

    @abstractmethod
    def set_value(self, value: float) -> None:
        """Set the value of this data provider.
        Raises exception if this data provider is read-only
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_data(self) -> Union[xr.DataArray, xr.Dataset]:
        """Returns the raw data backing this provider as an
        xarray.DataArray
        """
        raise NotImplementedError
