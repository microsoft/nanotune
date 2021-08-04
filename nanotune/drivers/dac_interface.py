from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type
from enum import Enum

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import ChannelList, InstrumentChannel


class RelayState(Enum):
    """Possible relay states of a DAC channel."""
    ground = 0
    smc = 1
    bus = 2
    floating = 3


class DACChannelInterface(InstrumentChannel, ABC):
    """A DAC channel interface. By inheriting from this class and implementing
    all abstract methods, the user specifies how the DAC and its channels are
    set or gotten. It allows to keep the rest of nanotune independent of the
    instruments used.

    Properties:
        channel_id: The number of the channel on the DAC instrument.
        supports_hardware_ramp: whether the channel's voltage can be ramped by
        using a hardware ramp.
    """

    def __init__(self, parent, name, channel_id):
        super().__init__(parent, name)
        self._channel_id = channel_id

    @property
    def channel_id(self) -> int:
        """The channel number. E.g. for dac.ch04, channel_id = 4."""
        return self._channel_id

    @property
    @abstractmethod
    def supports_hardware_ramp(self) -> bool:
        """Should indicate whether the channel's voltage can be ramped by
        using a hardware ramp.
        """
        pass

    @abstractmethod
    def set_voltage(self, new_voltage: float) -> None:
        """The channel's voltage setting method."""
        pass

    @abstractmethod
    def get_voltage(self) -> float:
        """The channel's voltage getting method."""
        pass

    @abstractmethod
    def set_voltage_limit(self, new_limits: Tuple[float, float]) -> None:
        """Sets the safe min and max voltages of the channel."""
        pass

    @abstractmethod
    def get_voltage_limit(self) -> Tuple[float, float]:
        """Gets the safe min and max voltages of the channel."""
        pass

    @abstractmethod
    def get_voltage_step(self) -> float:
        """Gets the `step` attribute of the voltage parameter."""
        pass

    @abstractmethod
    def set_voltage_step(self, new_step: float) -> None:
        """Sets the `step` attribute of the voltage parameter."""
        pass

    @abstractmethod
    def get_frequency(self) -> float:
        """Gets the channel's frequency if the DAC supports AWG
        functionalities.
        """
        pass

    @abstractmethod
    def set_frequency(self, value: float) -> None:
        """Sets the channel's frequency if the DAC supports AWG
        functionalities.
        """
        pass

    @abstractmethod
    def get_offset(self) -> float:
        """Gets the channel's waveform offset if the DAC supports AWG
        functionalities.
        """
        pass

    @abstractmethod
    def set_offset(self, value: float):
        """Sets the channel's waveform offset if the DAC supports AWG
        functionalities.
        """
        pass

    @abstractmethod
    def get_amplitude(self) -> float:
        """Gets the channel's waveform amplitude if the DAC supports AWG
        functionalities.
        """
        pass

    @abstractmethod
    def set_amplitude(self, value: float):
        """Sets the channel's waveform amplitude if the DAC supports AWG
        functionalities.
        """
        pass

    @abstractmethod
    def get_relay_state(self) -> RelayState:
        """Gets's the channel's relay state. E.g. float, or grounded."""
        pass

    @abstractmethod
    def set_relay_state(self, new_state: RelayState):
        """Sets's the channel's relay state. E.g. float, or grounded.
        """
        pass

    @abstractmethod
    def ramp_voltage(self, target_voltage: float, ramp_rate: Optional[float] = None):
        """Ramps the channel's voltage to a target voltage at either a
        specified ramp rate or for example self.get_ramp_rate()."""
        pass

    @abstractmethod
    def set_ramp_rate(self, value: float):
        """Sets the channel's voltage ramp rate."""
        pass

    @abstractmethod
    def get_ramp_rate(self) -> float:
        """Get channel's voltage ramp rate"""
        pass

    @abstractmethod
    def get_waveform(self) -> str:
        """Get channel's waveform if AWG functionalities are supported."""
        pass

    @abstractmethod
    def set_waveform(self, waveform: str):
        """Set channel's waveform if AWG functionalities are supported."""
        pass


class DACInterface(Instrument):
    """Interface for a DAC instrument defining the name of its channel list and
    how each individual channel is called. Its subclass should be used in
    measurements.
    """
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
        """Starts waveforms if AWG functionalities are supported."""
        pass

    @abstractmethod
    def sync(self):
        """Syncs the instrument."""
        pass
