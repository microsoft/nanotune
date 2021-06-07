# pylint: disable=too-many-arguments, too-many-locals

import os
from typing import Any, Optional, Sequence, Union

import qcodes as qc
import xarray as xr
from scipy import interpolate

from sim.data_providers import DataProvider


class StaticDataProvider(DataProvider):
    """Data provider that returns a constant value for all inputs."""

    @classmethod
    def make(cls, **kwargs) -> Any:
        """ ISerializable override to create an instance of this class """
        return cls(kwargs["value"])

    def __init__(self, value: float) -> None:
        super().__init__(settable=True)
        self._value = value

    def __call__(self, *args) -> float:
        return self._value

    def get_value(self) -> float:
        """The current value of this data provider"""

        return self._value

    def set_value(self, value: float) -> None:
        """Set the static value of this data provider"""
        self._value = value

    @property
    def raw_data(self) -> xr.DataArray:
        """Returns the raw data backing this provider as an
        xarray.DataArray
        """

        return xr.DataArray("0", dims="x", coords={"x": [1]})
