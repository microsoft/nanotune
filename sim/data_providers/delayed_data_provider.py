# pylint: disable=too-many-arguments, too-many-locals

from typing import Optional, Union

import xarray as xr

from sim.data_provider import IDataProvider, DataProvider
from sim.data_providers import StaticDataProvider
import time

class DelayedDataProvider(DataProvider):
    """ Data provider that wraps another provider to introduce a simple delay when
        reading or writing from another data provider """

    @classmethod
    def make(cls, **kwargs):
        raise NotImplementedError

    def __init__(
            self,
            read_delay : float,
            write_delay : float,
            child_data_provider : IDataProvider = StaticDataProvider(0.0)
        ):
        self._read_delay = read_delay
        self._write_delay = write_delay
        self._child_provider = child_data_provider
        super().__init__(settable = child_data_provider.settable)

    def get_value(self) -> float:
        time.sleep(self._read_delay)
        return self._child_provider.get_value()

    def set_value(self, value : float) -> None:
        time.sleep(self._write_delay)
        self._child_provider.set_value(value)

    def set_delays(
            self, read_delay : Optional[float] = None,
            write_delay : Optional[float] = None
        ) -> None:
        if read_delay:
            self._read_delay = read_delay

        if write_delay:
            self._write_delay = write_delay

    def set_data_provider(self, child_data_provider : IDataProvider) -> None:
        self._child_provider = child_data_provider

    @property
    def raw_data(self) -> Union[xr.DataArray, xr.Dataset]:
        return self._child_provider.raw_data
