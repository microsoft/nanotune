from nanotune.drivers.dac_interface import DACChannelInterface, RelayState
from typing import Optional, Tuple


class DACChannelServer(DACChannelInterface):
    def __init__(self, parent, name, channel_id):
        super().__init__(parent, name, channel_id)
        self._voltage_limit = (-3, 0)

    def supports_hardware_ramp(self) -> bool:
        # val = self.parent.send(self._get_msg("ramp"))
        # return bool(self._parse_return(val))
        return False

    def name(self) -> str:
        return self._name

    def set_voltage(self, new_val):
        self.parent.send(self._set_msg("voltage", new_val))

    def get_voltage(self):
        val = self.parent.send(self._get_msg("voltage"))
        return float(self._parse_return(val))

    def set_voltage_limit(self, new_limits: Tuple[float, float]) -> None:
        self._voltage_limit = new_limits

    def get_voltage_limit(self) -> Tuple[float, float]:
        return self._voltage_limit

    def get_voltage_step(self) -> float:
        val = self.parent.send(self._get_msg("step"))
        return float(self._parse_return(val))

    def set_voltage_step(self, new_step) -> None:
        self.parent.send(self._set_msg("step", new_step))

    def get_frequency(self) -> float:
        val = self.parent.send(self._get_msg("frequency"))
        return float(self._parse_return(val))

    def set_frequency(self, value) -> None:
        self.parent.send(self._set_msg("frequency", value))

    def get_offset(self) -> float:
        val = self.parent.send(self._get_msg("offset"))
        return float(self._parse_return(val))

    def set_offset(self, new_offset) -> None:
        self.parent.send(self._set_msg("offset", new_offset))

    def get_amplitude(self) -> float:
        val = self.parent.send(self._get_msg("amplitude"))
        return float(self._parse_return(val))

    def set_amplitude(self, new_amplitude) -> None:
        self.parent.send(self._set_msg("amplitude", new_amplitude))

    def get_relay_state(self) -> RelayState:
        raise NotImplementedError

    def set_relay_state(self, new_state: RelayState):
        raise NotImplementedError

    def _get_msg(self, param) -> str:
        return f"{self.channel_id};{param};?"

    def _set_msg(self, param, value) -> str:
        return f"{self.channel_id};{param};{value}"

    def ramp_voltage(self, target_voltage: float, ramp_rate: Optional[float] = None):
        self.set_voltage(target_voltage)

    def set_ramp_rate(self, value: float):
        raise NotImplementedError

    def get_ramp_rate(self) -> float:
        raise NotImplementedError

    def get_waveform(self) -> str:
        raise NotImplementedError

    def set_waveform(self, waveform: str):
        raise NotImplementedError
