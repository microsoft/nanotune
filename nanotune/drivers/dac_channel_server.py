from nanotune.drivers.dac_interface import DACChannelInterface


class DACChannelServer(DACChannelInterface):
    def __init__(self, parent, name, channel_id):
        super().__init__(parent, name, channel_id)

    def supports_ramp(self) -> bool:
        val = self.parent.send(self._get_msg("ramp"))
        return bool(self._parse_return(val))

    def name(self) -> str:
        return self._name

    def _set_dc_voltage(self, new_val):
        self.parent.send(self._set_msg("voltage", new_val))

    def _get_dc_voltage(self):
        val = self.parent.send(self._get_msg("voltage"))
        return float(self._parse_return(val))

    def _get_inter_delay(self) -> float:
        val = self.parent.send(self._get_msg("inter_delay"))
        return float(self._parse_return(val))

    def _set_inter_delay(self, new_inter_delay) -> None:
        self.parent.send(self._set_msg("inter_delay", new_inter_delay))

    def _get_post_delay(self) -> float:
        val = self.parent.send(self._get_msg("post_delay"))
        return float(self._parse_return(val))

    def _set_post_delay(self, new_post_delay) -> None:
        self.parent.send(self._set_msg("post_delay", new_post_delay))

    def _get_step(self) -> float:
        val = self.parent.send(self._get_msg("step"))
        return float(self._parse_return(val))

    def _set_step(self, new_step) -> None:
        self.parent.send(self._set_msg("step", new_step))

    def _get_label(self) -> str:
        val = self.parent.send(self._get_msg("frequency"))
        return self._parse_return(val)

    def _set_label(self, new_label) -> None:
        self.parent.send(self._set_msg("label", new_label))

    def _get_frequency(self) -> float:
        val = self.parent.send(self._get_msg("frequency"))
        return float(self._parse_return(val))

    def _set_frequency(self, new_frequency) -> None:
        self.parent.send(self._set_msg("frequency", new_frequency))

    def _get_offset(self, attr):
        val = self.parent.send(self._get_msg("offset"))
        return float(self._parse_return(val))

    def _set_offset(self, new_offset) -> None:
        self.parent.send(self._set_msg("offset", new_offset))

    def _get_amplitude(self, attr) -> float:
        val = self.parent.send(self._get_msg("amplitude"))
        return float(self._parse_return(val))

    def _set_amplitude(self, new_amplitude) -> None:
        self.parent.send(self._set_msg("amplitude", new_amplitude))

    def _get_relay_state(self, attr) -> int:
        raise NotImplementedError

    def _set_relay_state(self, value) -> None:
        raise NotImplementedError

    def _get_msg(self, param) -> str:
        return f"{self.channel_id};{param};?"

    def _set_msg(self, param, value) -> str:
        return f"{self.channel_id};{param};{value}"

    def _parse_return(self, value) -> str:
        if value == "None" or value is None:
            value = "0"
        return value
