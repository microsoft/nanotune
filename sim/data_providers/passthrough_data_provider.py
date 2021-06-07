# pylint: disable=too-many-arguments, too-many-locals

import os
from typing import Any, Optional, Sequence, Union

import qcodes as qc
import xarray as xr
from scipy import interpolate

from sim.data_providers import DataProvider
from sim.mock_device_registry import MockDeviceRegistry
from sim.mock_pin import IMockPin

class PassthroughDataProvider(DataProvider):
    """ Data provider that simply passes through a value from another input """

    @classmethod
    def make(cls, **kwargs):
        src_pin = MockDeviceRegistry.resolve_pin(kwargs["src_pin"])
        return cls(src_pin)

    def __init__(self, src_pin : IMockPin):
        super().__init__(settable = False)
        self._source = src_pin

    def get_value(self) -> float:
        return self._source.get_value()

    def set_value(self, value : float) -> None:
        self._source.set_value(value)

    @property
    def raw_data(self) -> Union[xr.DataArray, xr.Dataset]:
        return super().raw_data
