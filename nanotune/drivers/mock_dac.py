# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import Optional, Tuple
from nanotune.drivers.dac_interface import DACChannelInterface, DACInterface


class MockDACChannel(DACChannelInterface):
    def __init__(self, parent, name, channel_id):
        super().__init__(parent, name, channel_id)

        self._gnd: str = "close"
        self._bus: str = "close"
        self._smc: str = "close"
        self._dac_output: str = "close"

        self._label = "mock channel"
        self._curr_voltage = 0
        self._dc_voltage_limit = [-5, 5]
        self._inter_delay = 0.01
        self._step = 0
        self._limit_rate = 0.2
        self._relay_state = "smc"
        self._filter = 1
        self._waveform = "saw"
        self._amplitude = 0
        self._offset = 0
        self._frequency = 0
        self._period = 1
        self._phase = 0

    @property
    def supports_hardware_ramp(self) -> bool:
        return False

    def set_dc_voltage(self, new_voltage: float) -> None:
        self._curr_voltage = new_voltage

    def get_dc_voltage(self) -> float:
        return self._curr_voltage

    def set_dc_voltage_limit(self, new_limits: Tuple[float, float]) -> None:
        self._dc_voltage_limit = new_limits

    def get_inter_delay(self) -> float:
        return self._inter_delay

    def set_inter_delay(self, new_inter_delay: float) -> None:
        self._inter_delay = new_inter_delay

    def get_post_delay(self) -> float:
        return self._post_delay

    def set_post_delay(self, new_post_delay: float) -> None:
        self._post_delay = new_post_delay

    def get_step(self) -> float:
        return self._step

    def set_step(self, new_step: float) -> None:
        self._step = new_step

    def get_label(self) -> str:
        return self._label

    def set_label(self, new_label: str) -> None:
        self._label = new_label

    def get_frequency(self) -> float:
        return self._frequency

    def set_frequency(self, new_frequency: float) -> None:
        self._frequency = new_frequency

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, new_offset: float):
        self._offset = new_offset

    def get_amplitude(self) -> float:
        return self._amplitude

    def set_amplitude(self, new_amplitude: float) -> None:
        self._amplitude = new_amplitude

    def get_relay_state(self) -> str:
        return self._relay_state

    def set_relay_state(self, new_relay_state: str) -> None:
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
        self._curr_voltage = target_voltage


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