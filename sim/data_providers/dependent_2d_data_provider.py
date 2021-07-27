# pylint: disable=too-many-arguments, too-many-locals

import os
from typing import Any, Callable, Optional, Union

import qcodes as qc
import xarray as xr
from scipy import interpolate

from sim.data_provider import DataProvider
from sim.mock_device_registry import MockDeviceRegistry
from sim.mock_pin import IMockPin

class Dependent2dDataProvider(DataProvider):
    """ Read-only 2D Data provider that constructs it's value from two other data providers """

    @classmethod
    def make(cls, **kwargs):
        src1_pin = MockDeviceRegistry.resolve_pin(kwargs["src1_pin"])
        src2_pin = MockDeviceRegistry.resolve_pin(kwargs["src2_pin"])
        return cls(src1_pin, src2_pin)

    def __init__(self,
            src1_pin : IMockPin,
            src2_pin : IMockPin,
            value_provider : Optional[Callable[[float, float], float]] = None):
        super().__init__(settable = False)
        self._source1 = src1_pin
        self._source2 = src2_pin
        self._value_provider : Callable[[float, float], float] = \
            value_provider if value_provider else Dependent2dDataProvider.add_values

    def get_value(self) -> float:
        return self._value_provider(self._source1.get_value(), self._source2.get_value())

    def set_value(self, value : float) -> None:
        raise NotImplementedError

    @staticmethod
    def add_values(value1 : float, value2 : float) -> float:
        return value1 + value2

    @staticmethod
    def mul_values(value1 : float, value2 : float) -> float:
        return value1 * value2

    @property
    def raw_data(self) -> Union[xr.DataArray, xr.Dataset]:
        return super().raw_data
