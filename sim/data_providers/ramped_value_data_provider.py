import time
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from sim.data_provider import DataProvider
from sim.mock_pin import IMockPin
from sim.mock_device_registry import MockDeviceRegistry

class RampedValueDataProvider(DataProvider):
    """ Data provider that simulates value changes by ramping the value over time
        ramp_rate_provider supplies the ramp rate in terms of rate/min
    """

    @classmethod
    def make(cls, **kwargs):
        """ sim.ISerialization override to support loading from yaml """

        ramp_rate_provider : IMockPin = MockDeviceRegistry.resolve_pin(kwargs["ramp_rate_provider"])
        is_blocking_provider : IMockPin = MockDeviceRegistry.resolve_pin(kwargs["is_blocking_provider"])
        starting_value = kwargs.get("starting_value", 0.0)
        return cls(ramp_rate_provider, is_blocking_provider, starting_value)

    @classmethod
    def create(cls, starting_value : float, ramp_rate_per_min : float, is_blocking : bool):
        """ Helper to create a RampedValueDataProvider without the need to specify IMockPins for
            the ramp_rate and is_blocking parameters.  This is useful when you dont need to
            change the ramp rate or blocking behavior """

        from sim.mock_devices.pin import Pin

        ramp_rate_pin : IMockPin = Pin("ramp_rate")
        is_blocking_pin : IMockPin = Pin("blocking")
        ramp_rate_pin.set_value(ramp_rate_per_min)
        is_blocking_pin.set_value(1.0 if is_blocking else 0.0)

        return cls(ramp_rate_pin, is_blocking_pin, starting_value)

    def __init__(
            self,
            ramp_rate_provider : IMockPin,
            is_blocking_provider : IMockPin,
            starting_value : float
        ):
        """
        :param IMockPin ramp_rate_provider: Supplies the ramp rate to the data provider in terms of rate/min
        :param IMockPin is_blocking_provider: Informs the data provider whether to block until target value is reached
        :param float starting_value: Initial value
        """
        super().__init__(settable=True)

        self._value = starting_value
        self._ramp_rate = ramp_rate_provider
        self._blocking = is_blocking_provider
        self._ramp_start_time : Optional[float] = None
        self._wait_time = 0.0
        self._setCount = 0
        self._ramp_gen = self._value_ramp()
        next(self._ramp_gen)

    def get_value(self) -> float:
        """ Retrieve the current value """
        if self._ramp_start_time:
            _time_since_start = time.time() - self._ramp_start_time
            val = self._ramp_gen.send(_time_since_start)
            next(self._ramp_gen)
            self._value = val
        return self._value

    def set_value(self, value : float) -> None:
        """ Begins ramping to the specified value at the ramp rate defined by the ramp_rate_provider """

        if self._value != value:
            self._wait_time = 60. * np.abs(self._value - value) / self._ramp_rate.get_value()
            self._value_ramp_fcn = interp1d(
                x=[0.0, self._wait_time],
                y=[self._value, value],
                bounds_error=False,
                fill_value=(self._value, value)
            )
            self._ramp_start_time = time.time()

            if self._blocking and self._blocking.get_value():
                time.sleep(self._wait_time)
                self._value = value

    def _value_ramp(self):
        """ Generates the next value based on the elapsed time """
        while True:
            _time = yield
            if _time is None:
                _time = 0.0

            yield float(self._value_ramp_fcn(_time))

    def raw_data(self):
        raise NotImplementedError()
