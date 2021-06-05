# pylint: disable=too-many-arguments, too-many-locals

import os
from typing import Any, Optional, Sequence, Union
import numpy as np

import xarray as xr

from sim.data_providers import DataProvider
from sim.mock_device_registry import MockDeviceRegistry
from sim.mock_pin import IMockPin

class SyntheticPinchoffDataProvider(DataProvider):
    """ Simulates pinchoff curves using hyperbolic tangent funtion.

        Inputs:
        min, max   : controls the lower and upper bounds of output
        center     : Defines x value where the center of the curve will be located
        width      : Defines the overall width of the curve
        upper_tilt : If non-zero the upper half of the function will follow a
                     linear slope instead of flattening out at the max value
        lower_tilt : If non-zero the lower half of the function will follow a
                     linear slope instead of flattening out at the max value
        noise      : If non-zero, gaussian noise will be added. The value
                     indicates the amplitude of the noise in terms of percentage
                     of the height of the curve (max - min).
                     e.g.  0.03 would indicate 3% of the height of the curve
        flip       : If False, the curve will rise with increasing x values.
                     If True, the curve will fall with increasing x values.
    """

    @classmethod
    def make(cls, **kwargs):
        src_pin = MockDeviceRegistry.resolve_pin(kwargs["src_pin"])
        min : float = kwargs["min"]
        max : float = kwargs["max"]
        center : float = kwargs["center"]
        width : float = kwargs["width"]
        upper_tilt : float = kwargs.get("upper_tilt", 0.0)
        lower_tilt : float = kwargs.get("lower_tilt", 0.0)
        noise : float = kwargs.get("noise", 0.0)
        flip : bool = kwargs.get("flip", False)
        raw_data_samples : int = kwargs.get("raw_data_samples", 500)

        return cls(src_pin,
                   min, max,
                   center, width,
                   upper_tilt, lower_tilt,
                   flip,
                   raw_data_samples)

    def __init__(
            self,
            src_pin : IMockPin,
            min : float,
            max : float,
            center : float,
            width : float,
            upper_tilt : float = 0.0,
            lower_tilt : float = 0.0,
            noise : float = 0.0,  # % of height
            flip : bool = False,
            raw_data_samples : int = 500
        ):
        super().__init__(settable = False)
        self._source = src_pin
        self._min = min
        self._max = max
        self._center = center
        self._width = width
        self._upper_tilt = upper_tilt
        self._lower_tilt = lower_tilt
        self._noise = noise
        self._flip = flip
        self._raw_data_samples = raw_data_samples

    def get_value(self) -> float:
        return self._compute(self._source.get_value())

    def set_value(self, value : float) -> None:
        """Raises NotImplementedError.  This data provider type is read only"""
        raise NotImplementedError

    def _compute(self, x : float) -> float:
        """ Compute the result from a simulated pinchoff curve using the configured inputs """
        height : float = self._max - self._min
        sign : float = -1.0 if self._flip else 1.0

        a : float = 6.4 / self._width
        b : float = a * self._center
        c : float = 2 / height
        d : float = self._min + (0.5 * height)
        t : float = 0.0

        value = np.tanh((sign*a*x) - (sign*b))/c + d

        t = self._upper_tilt
        is_upper_half = ((x > self._center and not self._flip) or
                         (x < self._center and self._flip))

        if t and is_upper_half:
            value = value + (np.abs(x-self._center)/t)

        t = self._lower_tilt
        if t and not is_upper_half:
            value = value - (np.abs(x-self._center)/t)

        if (self._noise):
            value = value + np.random.normal() * height * self._noise

        return value

    @property
    def raw_data(self) -> Union[xr.DataArray, xr.Dataset]:
        """ Returns a snapshot of the function consisting of configured number
            of samples spanning the 2x the specified width around the center of
            the curve.
        """
        x = np.linspace(self._center - self._width, self._center + self._width, self._raw_data_samples)
        y = [self._compute(val) for val in x]
        return xr.DataArray(y, dims=["x"], coords={"x": x})
