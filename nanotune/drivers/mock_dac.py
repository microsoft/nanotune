# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import Optional, Tuple
from qcodes import validators as vals
from nanotune.drivers.dac_interface import (DACChannelInterface, DACInterface,
    RelayState)


class MockDACChannel(DACChannelInterface):
    def __init__(self, parent, name, channel_id):
        super().__init__(parent, name, channel_id)

        self._label = "mock channel"
        self._curr_voltage = 0
        self._voltage_limit = [-5, 5]
        self._step = 0
        self._limit_rate = 0.2
        self._relay_state: RelayState = RelayState.smc
        self._filter = 1
        self._waveform = "saw"
        self._amplitude = 0
        self._offset = 0
        self._frequency = 0
        self._period = 1
        self._phase = 0


        super().add_parameter(
            name="voltage",
            label=f"{name} dc voltage",
            set_cmd=self.set_voltage,
            get_cmd=self.get_voltage,
            vals=vals.Numbers(*self._voltage_limit),
            )

        super().add_parameter(
            name="amplitude",
            label=f"{name} amplitude",
            set_cmd=self.set_amplitude,
            get_cmd=self.get_amplitude,
            vals=vals.Numbers(),
            )

        super().add_parameter(
            name="frequency",
            label=f"{name} frequency",
            set_cmd=self.set_frequency,
            get_cmd=self.get_frequency,
            vals=vals.Numbers(),
        )

    @property
    def gettable(self):
        return False

    @property
    def settable(self):
        return False

    @property
    def supports_hardware_ramp(self) -> bool:
        return False

    def set_voltage(self, new_voltage: float) -> None:
        self._curr_voltage = new_voltage

    def get_voltage(self) -> float:
        return self._curr_voltage

    def set_voltage_limit(self, new_limits: Tuple[float, float]) -> None:
        self.voltage.vals = vals.Numbers(*new_limits)

    def get_voltage_limit(self) -> Tuple[float, float]:
        return self.voltage.vals.valid_values

    def get_voltage_inter_delay(self) -> float:
        return self.voltage.inter_delay

    def set_voltage_inter_delay(self, new_inter_delay: float) -> None:
        self.voltage.inter_delay = new_inter_delay

    def get_voltage_post_delay(self) -> float:
        return self.voltage.post_delay

    def set_voltage_post_delay(self, new_post_delay: float) -> None:
        self.voltage.post_delay = new_post_delay

    def get_voltage_step(self) -> float:
        return self.voltage.step

    def set_voltage_step(self, new_step: float) -> None:
        self.voltage.step = new_step

    def get_frequency(self) -> float:
        return self._frequency

    def set_frequency(self, value: float) -> None:
        self._frequency = value

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, new_offset: float):
        self._offset = new_offset

    def get_amplitude(self) -> float:
        return self._amplitude

    def set_amplitude(self, new_amplitude: float) -> None:
        self._amplitude = new_amplitude

    def get_relay_state(self) -> RelayState:
        return self._relay_state

    def set_relay_state(self, new_relay_state: RelayState) -> None:
        self._relay_state = new_relay_state

    def set_ramp_rate(self, new_ramp_rate: float) -> None:
        self._ramp_rate = new_ramp_rate

    def get_ramp_rate(self) -> float:
        return self._ramp_rate

    def get_limit_rate(self) -> float:
        return self._limit_rate

    def set_limit_rate(self, new_limit_rate: float) -> None:
        self._limit_rate = new_limit_rate

    def get_filter(self) -> int:
        return self._filter

    def set_filter(self, filter_id: int) -> None:
        self._filter = filter_id

    def get_waveform(self) -> str:
        return self._waveform

    def set_waveform(self, waveform: str) -> None:
        self._waveform = waveform

    def ramp_voltage(
        self, target_voltage: float, ramp_rate: Optional[float] = None
    ) -> None:
        raise NotImplementedError


class MockDAC(DACInterface):
    """
    Dummy instrument with nt channels
    """

    def __init__(self, name, DACChannelClass):
        super().__init__(name, DACChannelClass)

    def run(self) -> None:
        print("running mock dac")

    def sync(self) -> None:
        print("syncing mock dac")
