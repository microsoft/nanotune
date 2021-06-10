import copy
import logging
import time
from contextlib import contextmanager
from math import isclose
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import qcodes as qc
from qcodes import ArrayParameter, Instrument, InstrumentChannel, Parameter
from qcodes import validators as vals
from qcodes.instrument.base import InstrumentBase
from qcodes.utils.validators import Validator

import nanotune as nt
from nanotune.drivers.dac_interface import DACInterface

logger = logging.getLogger(__name__)


class DeviceChannel(InstrumentChannel):
    """"""

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        channel: Union[str, InstrumentChannel],
        gate_id: Optional[int] = None,
        ohmic_id: Optional[int] = None,
        label: str = "device_channel",
        safety_voltage_range: Tuple[float, float] = (-3, 0),
        use_ramp: bool = True,
        ramp_rate: float = 0.1,
        max_voltage_step: float = 0.05,
        post_delay: float = 0.001,
        inter_delay: float = 0.001,
        **kwargs,
    ) -> None:

        """

        Args:


        """

        self._channel = self._get_channel_instance(channel)

        if gate_id is not None and ohmic_id is not None:
            raise ValueError(f"Initializing {name} as both gate and ohmic.")

        channel_name = self._channel.name
        if channel_name.startswith(parent.name):
            channel_name = channel_name.replace(parent.name+"_", "", 1)

        static_metadata = kwargs.pop('metadata', {})
        static_metadata.update({
            'label': label,
            'gate_id':gate_id,
            'ohmic_id': ohmic_id}
        )
        assert issubclass(parent.__class__, DACInterface)

        super().__init__(
            parent,
            channel_name,
            metadata=static_metadata,
            **kwargs,
        )

        self._gate_id = gate_id
        self._ohmic_id = ohmic_id

        super().add_parameter(
            name="voltage",
            label=f"{label} voltage",
            set_cmd=self.set_voltage,
            get_cmd=self._channel.get_voltage,
            vals=vals.Numbers(*safety_voltage_range),
        )

        self.label = label
        self.inter_delay = inter_delay
        self.post_delay = post_delay

        super().add_parameter(
            name="safety_voltage_range",
            label=f"{label} voltage safety range",
            set_cmd=self._set_safety_voltage_range,
            get_cmd=self._get_safety_voltage_range,
            initial_value=list(safety_voltage_range),
            vals=vals.Lists(),
        )

        super().add_parameter(
            name="max_voltage_step",
            label=f"{label} maximum voltage jump",
            set_cmd=self._set_max_voltage_step,
            get_cmd=self._get_max_voltage_step,
            initial_value=max_voltage_step,
            vals=vals.Numbers(),
        )

        super().add_parameter(
            name="ramp_rate",
            label=f"{label} ramp rate",
            set_cmd=self._channel.set_ramp_rate,
            get_cmd=self._channel.get_ramp_rate,
            initial_value=ramp_rate,
            vals=vals.Numbers(),
        )

        super().add_parameter(
            name="use_ramp",
            label=f"{label} ramp setting",
            set_cmd=self._set_ramp,
            get_cmd=self._get_ramp,
            initial_value=use_ramp,
            vals=vals.Bool(),
        )

        super().add_parameter(
            name="relay_state",
            label=f"{label} DAC channel relay state",
            set_cmd=self._channel.set_relay_state,
            get_cmd=self._channel.get_relay_state,
            initial_value=self._channel.get_relay_state(),
            vals=vals.Strings(),
        )

        super().add_parameter(
            name="amplitude",
            label=f"{label} amplitude",
            set_cmd=self._channel.set_amplitude,
            get_cmd=self._channel.get_amplitude,
            initial_value=0,
            vals=vals.Numbers(-0.5, 0.5),
        )

        super().add_parameter(
            name="offset",
            label=f"{label} offset",
            set_cmd=self._set_offset,
            get_cmd=self._channel.get_offset,
            initial_value=0,
            vals=vals.Numbers(*safety_voltage_range),
        )

        super().add_parameter(
            name="frequency",
            label=f"{label} frequency",
            set_cmd=self._channel.set_frequency,
            get_cmd=self._channel.get_frequency,
            initial_value=0,
            vals=vals.Numbers(),
        )

    @property
    def gate_id(self) -> int:
        return self._gate_id

    @property
    def ohmic_id(self) -> int:
        return self._ohmic_id

    @property
    def has_ramp(self) -> float:
        return self._channel.supports_hardware_ramp

    @property
    def post_delay(self) -> float:
        return self._channel.get_voltage_post_delay()

    @post_delay.setter
    def post_delay(self, new_value: float) -> None:
        self._channel.set_voltage_post_delay(new_value)

    @property
    def inter_delay(self) -> float:
        return self._channel.get_voltage_inter_delay()

    @inter_delay.setter
    def inter_delay(self, new_value: float) -> None:
        self._channel.set_voltage_inter_delay(new_value)

    def ground(self) -> None:
        """ """
        self.relay_state("ground")

    def float_relay(self) -> None:
        """ """
        self.relay_state("float")

    def set_voltage(self, new_value: float, tol: float = 1e-5):
        """Voltage setter wich checks if new voltage is within safety ranges
        before setting anything - as opposed to qcodes, which throws an error
        only once the safety range is reached.
        It ramps to the new value if ramping is enabled, using either software
        or hardware ramps depending on `has_ramp`. It's a blocking ramp.

        Args:
            new_value: New voltage value to set.
            tol: Tolerance in Volt.
        """

        safe_range = self.safety_voltage_range()
        if new_value < safe_range[0] - tol or new_value > safe_range[1] + tol:
            raise ValueError(
                f"Setting voltage outside of permitted range: \
                    {self.label} to {new_value}.")

        current_range = self.current_valid_range()
        if new_value < current_range[0] - 1e-5 or new_value > current_range[0]:
            logger.debug(
                f"Setting {self.label}'s voltage outside of current valid \
                    range."
            )

        if self.has_ramp and self.use_ramp():
            self._channel.ramp_voltage(new_value)
            while not isclose(new_value, self.voltage(), abs_tol=tol):
                time.sleep(0.02)
            time.sleep(self.post_delay())
        elif not self.has_ramp and self.use_ramp():
            # Use qcodes software ramp
            step = self.max_voltage_step()
            with set_inter_delay(self.max_voltage_step(), self.ramp_rate()):
                self._channel.set_voltage(new_value)
        elif not self.use_ramp():
            self._channel.set_voltage(new_value)
        else:
            raise ValueError(
                "Unknown voltage setting mode. Should I ramp or not?"
            )

    def _get_safety_voltage_range(self) -> List[float]:
        return copy.deepcopy(self._safety_voltage_range)

    def _set_safety_voltage_range(self, new_range: List[float]) -> None:
        assert len(new_range) == 2
        self._safety_voltage_range = sorted(new_range)
        logger.info(f"{self.name}: changing safety range to {new_range}")

        self._channel.set_voltage_limit(new_range)
        self.voltage.vals = vals.Numbers(*new_range)

    def _set_offset(self, new_offset: Optional[float]) -> None:
        safe_range = self.safety_voltage_range()
        if (new_offset >= safe_range[0] and new_offset <= safe_range[1]):
            self._channel.set_offset(new_offset)
        else:
            raise ValueError(
                f"{self.label}: offset voltage outside of safety range."
            )

    def _get_ramp(self):
        return self._ramp

    def _set_ramp(self, on: bool) -> None:
        if not on:
            logger.warning(
                f"{self.name} will not ramp voltages. Make sure to \
                choose a reasonable `max_voltage_step`."
            )
        self._ramp = on

    def _get_max_voltage_step(self) -> float:
        return self._channel.get_voltage_step()

    def _set_max_voltage_step(self, max_voltage_step: float) -> None:
        self._channel.set_voltage_step(max_voltage_step)

    def _get_channel_instance(
        self,
        channel: Union[str, InstrumentChannel],
    ) -> InstrumentChannel:
        if isinstance(channel, str):
            _ , channel_name = channel.split('.')
            instr_channel = getattr(parent, channel_name)
        elif isinstance(channel, InstrumentChannel):
            instr_channel = channel
        else:
            raise ValueError(f'Invalid type for "channel" input of {name}.')
        return instr_channel

    @contextmanager
    def set_inter_delay(max_voltage_step: float, ramp_rate: float):
        initial_rate = self._channel.get_voltage_inter_delay()
        init_step = self._channel.get_voltage_step()
        self._channel.set_voltage_inter_delay(max_voltage_step / ramp_rate)
        self._channel.set_voltage_step(max_voltage_step)
        try:
            yield
        finally:
            self._channel.set_voltage_inter_delay(initial_rate)
            self._channel.set_voltage_step(init_step)