from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type
from enum import Enum

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import ChannelList, InstrumentChannel


class RelayState(Enum):
    ground = 0
    smc = 1
    bus = 2
    floating = 3


class DACChannelInterface(InstrumentChannel, ABC):
    def __init__(self, parent, name, channel_id):
        super().__init__(parent, name)
        self._channel_id = channel_id

    @property
    def channel_id(self) -> int:
        return self._channel_id

    @property
    @abstractmethod
    def supports_hardware_ramp(self) -> bool:
        pass

    @abstractmethod
    def set_voltage(self, new_voltage: float) -> None:
        pass

    @abstractmethod
    def get_voltage(self) -> float:
        pass

    @abstractmethod
    def set_voltage_limit(self, new_limits: Tuple[float, float]) -> None:
        pass

    @abstractmethod
    def get_voltage_limit(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_voltage_step(self) -> float:
        pass

    @abstractmethod
    def set_voltage_step(self, new_step: float) -> None:
        pass

    @abstractmethod
    def get_frequency(self) -> float:
        pass

    @abstractmethod
    def set_frequency(self, new_frequency: float) -> None:
        pass

    @abstractmethod
    def get_offset(self) -> float:
        pass

    @abstractmethod
    def set_offset(self, value: float):
        pass

    @abstractmethod
    def get_amplitude(self) -> float:
        pass

    @abstractmethod
    def set_amplitude(self, value: float):
        pass

    @abstractmethod
    def get_relay_state(self) -> RelayState:
        pass

    @abstractmethod
    def set_relay_state(self, new_state: RelayState):
        """ """
        pass

    @abstractmethod
    def ramp_voltage(self, target_voltage: float, ramp_rate: Optional[float] = None):
        pass

    @abstractmethod
    def set_ramp_rate(self, value: float):
        pass

    @abstractmethod
    def get_ramp_rate(self) -> float:
        pass

    @abstractmethod
    def get_waveform(self) -> str:
        pass

    @abstractmethod
    def set_waveform(self, waveform: str):
        pass


class DACInterface(Instrument):
    def __init__(self, name, DACChannelClass: Type[DACChannelInterface]):
        assert issubclass(DACChannelClass, DACChannelInterface)
        super().__init__(name)

        channels = ChannelList(self, "Channels", DACChannelClass, snapshotable=False)
        for chan_id in range(0, 64):
            chan_name = f"ch{chan_id:02d}"
            channel = DACChannelClass(self, chan_name, chan_id)
            channels.append(channel)
            self.add_submodule(chan_name, channel)
        self.add_submodule("channels", channels)

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def sync(self):
        pass
