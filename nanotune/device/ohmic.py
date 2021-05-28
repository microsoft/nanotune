import logging
from typing import Any, Dict, Optional, Sequence

from qcodes import Instrument, InstrumentChannel
from qcodes import validators as vals
from qcodes.instrument.base import InstrumentBase

from nanotune.drivers.dac_interface import DACInterface

logger = logging.getLogger(__name__)


class Ohmic(InstrumentChannel):
    """
    ohmic_id: Specific to device, ie. fivedot - left ohmic is #1,
                counting clock wise to ohmic #10.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        dac_instrument: DACInterface,
        channel_id: int,
        ohmic_id: int,
        name: str = "ohmic",
        label: str = "ohmic",
        **kwargs,
    ) -> None:

        super().__init__(parent, name)

        self._dac_instrument = dac_instrument
        self._dac_channel = self._dac_instrument.nt_channels[channel_id]

        super().add_parameter(
            name="channel_id",
            label=" instrument channel id",
            set_cmd=None,
            get_cmd=None,
            initial_value=channel_id,
            vals=vals.Ints(0),
        )

        super().add_parameter(
            name="label",
            label="ohmic label",
            set_cmd=self._dac_channel.set_label,
            get_cmd=self._dac_channel.get_label,
            initial_value=label,
            vals=vals.Strings(),
        )

        super().add_parameter(
            name="ohmic_id",
            label=self.label() + " ohmic number",
            set_cmd=None,
            get_cmd=None,
            initial_value=ohmic_id,
            vals=vals.Ints(0),
        )

        super().add_parameter(
            name="relay_state",
            label=f"{label} DAC channel relay state",
            set_cmd=self._dac_channel.set_relay_state,
            get_cmd=self._dac_channel.get_relay_state,
            initial_value=self._dac_channel.get_relay_state(),
            vals=vals.Strings(),
        )

    def ground(self) -> None:
        """ """
        self.relay_state("ground")

    def float_relay(self) -> None:
        """ """
        self.relay_state("float")

    def snapshot_base(
        self,
        update: Optional[bool] = True,
        params_to_skip_update: Optional[Sequence[str]] = None,
    ) -> Dict[Any, Any]:
        """
        Add instrument source name to snapshot for reference
        """
        snap = super().snapshot_base(update, params_to_skip_update)
        snap["dac_channel"] = self._dac_channel.name
        return snap
