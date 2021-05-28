import gc
from typing import Optional, Tuple

import joblib
import pytest
from qcodes import new_experiment
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import ChannelList, InstrumentChannel
from qcodes.tests.instrument_mocks import (DummyChannel,
                                           DummyChannelInstrument,
                                           DummyInstrument,
                                           DummyInstrumentWithMeasurement)
from qcodes.utils import validators as vals

import nanotune as nt
from nanotune.classification.classifier import Classifier
from nanotune.device.device import Device
from nanotune.drivers.dac_interface import DACChannelInterface, DACInterface


class DummyDACChannel(DACChannelInterface):
    def __init__(self, parent, name, channel_id):
        super().__init__(parent, name, channel_id)

        self._gnd: str = "close"
        self._bus: str = "close"
        self._smc: str = "close"
        self._dac_output: str = "close"

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

    def set_dc_voltage(self, val) -> None:
        self._curr_voltage = val

    def get_dc_voltage(self) -> float:
        return self._curr_voltage

    def set_dc_voltage_limit(self, val: Tuple[float, float]) -> None:
        self._dc_voltage_limit = val

    def get_inter_delay(self) -> float:
        return self._inter_delay

    def set_inter_delay(self, val: float) -> None:
        self._inter_delay = val

    def get_post_delay(self) -> float:
        return self._post_delay

    def set_post_delay(self, val: float) -> None:
        self._post_delay = val

    def get_step(self) -> float:
        return self._step

    def set_step(self, val: float) -> None:
        self._step = val

    def get_label(self) -> str:
        return self._label

    def set_label(self, val: str) -> None:
        self._label = val

    def get_frequency(self) -> float:
        return self._frequency

    def set_frequency(self, val: float) -> None:
        self._frequency = val

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, val: float) -> None:
        self._offset = val

    def get_amplitude(self) -> float:
        return self._amplitude

    def set_amplitude(self, val: float) -> None:
        self._amplitude = val

    def get_relay_state(self) -> str:
        return self._relay_state

    def set_relay_state(self, val: str) -> None:
        self._relay_state = val

    def set_ramp_rate(self, val: float) -> None:
        self._ramp_rate = val

    def get_ramp_rate(self) -> float:
        return self._ramp_rate

    def get_limit_rate(self) -> float:
        return self._limit_rate

    def set_limit_rate(self, val: float) -> None:
        self._limit_rate = val

    def get_filter(self) -> int:
        return self._filter

    def set_filter(self, val: int) -> None:
        self._filter = val

    def get_waveform(self) -> str:
        return self._waveform

    def set_waveform(self, val: str) -> None:
        self._waveform = val

    def ramp_voltage(
        self, target_voltage: float, ramp_rate: Optional[float] = None
    ) -> None:
        self._curr_voltage = target_voltage


class DummyDAC(DACInterface):
    """
    Dummy instrument with nt channels
    """

    def __init__(self, name, DACChannelClass):
        super().__init__(name, DACChannelClass)

    def run(self) -> None:
        print("running dummy dac")

    def sync(self) -> None:
        print("syncing dummy dac")
