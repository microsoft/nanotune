import logging
import time
import json
import copy
from math import isclose
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any, Sequence

import qcodes as qc
from qcodes import validators as vals
from qcodes.utils.validators import Validator
from qcodes.instrument.base import InstrumentBase
from qcodes import Instrument, InstrumentChannel, Parameter, ArrayParameter

import nanotune as nt
from nanotune.drivers.dac_interface import DACInterface

logger = logging.getLogger(__name__)


class Gate(InstrumentChannel):
    """"""

    def __init__(
        self,
        parent: InstrumentBase,
        dac_instrument: DACInterface,
        channel_id: int,
        layout_id: int,
        name: str = "nanotune_gate",
        label: str = "nanotune gate",
        line_type: str = "dc_current",
        safety_range: Tuple[float, float] = (-3, 0),
        use_ramp: bool = True,
        ramp_rate: float = 0.1,
        max_jump: float = 0.05,
        delay: float = 0.001,
        fast_readout_source: Optional[InstrumentChannel] = None,
        **kwargs,
    ) -> None:

        """

        Args:
            dac_instrument (Optional[Parameter]):
            line_type (Optional[str]):
            min_v (Optional[float]):
            max_v (Optional[float]):
            use_ramp (Optional[bool]):
            ramp_rate (Optional[float]):
            max_jump (Optional[float]):
            delay (Optional[float]): Delay in between setting and measuring
                another parameter

        """
        assert issubclass(dac_instrument.__class__, DACInterface)

        super().__init__(parent, name)

        self._dac_instrument = dac_instrument
        self._dac_channel = self._dac_instrument.nt_channels[channel_id]

        super().add_parameter(
            name="channel_id",
            label="instrument channel id",
            set_cmd=None,
            get_cmd=None,
            initial_value=channel_id,
            vals=vals.Ints(0),
        )

        super().add_parameter(
            name="dc_voltage",
            label=f"{label} dc voltage",
            set_cmd=self.set_dc_voltage,
            get_cmd=self._dac_channel.get_dc_voltage,
            vals=vals.Numbers(*safety_range),
        )

        self.has_ramp = self._dac_channel.supports_hardware_ramp

        super().add_parameter(
            name="label",
            label="gate label",
            set_cmd=self._dac_channel.set_label,
            get_cmd=self._dac_channel.get_label,
            initial_value=label,
            vals=vals.Strings(),
        )

        qc_safety_range = (safety_range[0] - 0.01, safety_range[1] + 0.01)

        super().add_parameter(
            name="safety_range",
            label=f"{label} DC voltage safety range",
            set_cmd=self.set_safety_range,
            get_cmd=self.get_safety_range,
            initial_value=list(safety_range),
            vals=vals.Lists(),
        )

        super().add_parameter(
            name="inter_delay",
            label=f"{label} inter delay",
            set_cmd=self._dac_channel.set_inter_delay,
            get_cmd=self._dac_channel.get_inter_delay,
            initial_value=delay,
            vals=vals.Numbers(),
        )

        super().add_parameter(
            name="post_delay",
            label=label + " post delay",
            set_cmd=self._dac_channel.set_post_delay,
            get_cmd=self._dac_channel.get_post_delay,
            initial_value=delay,
            vals=vals.Numbers(),
        )

        super().add_parameter(
            name="max_jump",
            label=f"{label} maximum voltage jump",
            set_cmd=self.set_max_jump,
            get_cmd=self.get_max_jump,
            initial_value=max_jump,
            vals=vals.Numbers(),
        )

        super().add_parameter(
            name="step",
            label=f"{label} voltage step",
            set_cmd=self._dac_channel.set_step,
            get_cmd=self._dac_channel.get_step,
            initial_value=max_jump,
            vals=vals.Numbers(),
        )

        super().add_parameter(
            name="ramp_rate",
            label=f"{label} ramp rate",
            set_cmd=self._dac_channel.set_ramp_rate,
            get_cmd=self._dac_channel.get_ramp_rate,
            initial_value=ramp_rate,
            vals=vals.Numbers(),
        )

        super().add_parameter(
            name="use_ramp",
            label=f"{label} ramp setting",
            set_cmd=self.set_ramp,
            get_cmd=self.get_ramp,
            initial_value=use_ramp,
            vals=vals.Bool(),
        )

        super().add_parameter(
            name="relay_state",
            label=f"{label} DAC channel relay state",
            set_cmd=self._dac_channel.set_relay_state,
            get_cmd=self._dac_channel.get_relay_state,
            initial_value=self._dac_channel.get_relay_state(),
            vals=vals.Strings(),
        )

        super().add_parameter(
            name="layout_id",
            label=f"{label} device layout id",
            set_cmd=None,
            get_cmd=None,
            initial_value=layout_id,
            vals=vals.Ints(0),
        )

        super().add_parameter(
            name="current_valid_range",
            label=f"{label} current valid range",
            set_cmd=self.set_current_valid_range,
            get_cmd=self.get_current_valid_range,
            initial_value=[],
            vals=vals.Lists(),
        )

        super().add_parameter(
            name="transition_voltage",
            label=f"{label} transition voltage",
            set_cmd=self.set_transition_voltage,
            get_cmd=self.compute_transition_voltage,
            initial_value=0,
            vals=vals.Numbers(),
        )

        super().add_parameter(
            name="amplitude",
            label=f"{label} sawtooth amplitude",
            set_cmd=self._dac_channel.set_amplitude,
            get_cmd=self._dac_channel.get_amplitude,
            initial_value=0,
            vals=vals.Numbers(-0.5, 0.5),
        )

        super().add_parameter(
            name="offset",
            label=f"{label} sawtooth offset",
            set_cmd=self._set_offset,
            get_cmd=self._dac_channel.get_offset,
            initial_value=0,
            vals=vals.Numbers(*qc_safety_range),
        )

        super().add_parameter(
            name="frequency",
            label=f"{label} sawtooth frequency",
            set_cmd=self._dac_channel.set_frequency,
            get_cmd=self._dac_channel.get_frequency,
            initial_value=0,
            vals=vals.Numbers(),
        )

    def ground(self) -> None:
        """ """
        self.relay_state('ground')

    def float_relay(self) -> None:
        """ """
        self.relay_state('float')

    def snapshot_base(
        self,
        update: Optional[bool] = True,
        params_to_skip_update: Optional[Sequence[str]] = None,
    ) -> Dict[Any, Any]:
        """
        Add dc voltage source name to snapshot for reference.
        """
        snap = super().snapshot_base(update, params_to_skip_update)
        snap["dac_channel"] = self._dac_channel.name
        return snap

    def set_dc_voltage(self, new_value):
        """
        It will check if new voltage is within safety ranges before setting
        anything - as opposed to qcodes, which throws an error only once the
        safety range is reached.
        It ramp to the new value.
        """
        safe_range = self.safety_range()
        if new_value < safe_range[0] - 1e-5 or new_value > safe_range[1] + 1e-5:
            logger.error(
                ("Setting voltage out of permitted range: "
                f"{self.name} to {new_value}")
            )
            raise ValueError

        current_valid_range = self.current_valid_range()
        if new_value < current_valid_range[0] - 1e-5:
            current_valid_range[0] = new_value - 0.05
        if new_value > current_valid_range[1] + 1e-5:
            current_valid_range[1] = new_value + 0.05
        self.current_valid_range(current_valid_range)

        if self.has_ramp and self.use_ramp():
            self._dac_channel.ramp_voltage(new_value)
            while not isclose(new_value, self.dc_voltage(), abs_tol=1e-4):
                time.sleep(0.05)
            time.sleep(self.post_delay())
        elif not self.has_ramp and self.use_ramp():
            # Use qcodes software ramp
            step = self.max_jump()
            self.step(step)
            self.inter_delay(self.step() / self.ramp_rate())
            self._dac_channel.set_dc_voltage(new_value)

        else:
            self._dac_channel.set_dc_voltage(new_value)

    def get_safety_range(self) -> List[float]:
        """"""
        return copy.deepcopy(self._safety_range)

    def set_safety_range(self, new_range: List[float]) -> None:
        self._safety_range = new_range
        logger.info(f"{self.name}: changing safety range to {new_range}")
        # pass to underlying instrument channel safety range
        qc_safety_range = [0.0, 0.0]
        qc_safety_range[0] = new_range[0] - 0.01
        qc_safety_range[1] = new_range[1] + 0.01

        self._dac_channel.set_dc_voltage_limit(qc_safety_range)
        self.dc_voltage.vals = vals.Numbers(*qc_safety_range)

    def _set_offset(self, new_offset: Optional[float]) -> None:
        """ """
        if (
            new_offset >= self.safety_range()[0]
            and new_offset <= self.safety_range()[1]
        ):
            self._dac_channel.set_offset(new_offset)
        else:
            logger.error(
                f"Gate {self.label}: " "Invalid offset voltage. Keeping old one."
            )
            raise ValueError

    def get_ramp(self):
        return self._ramp

    def set_ramp(self, on: bool) -> None:
        if not on:
            logger.debug(
                f"{self.name} will not ramp voltages. Make sure to "
                "choose a reasonable stepsize."
            )
        self._ramp = on

    def get_max_jump(self) -> float:
        return self._max_jump

    def set_max_jump(self, max_jump: float) -> None:
        self._max_jump = max_jump

    def get_current_valid_range(self) -> List[float]:
        """"""
        return self._current_valid_range

    def set_current_valid_range(
        self,
        new_range: Union[Tuple[float, float], List[float]],
        ) -> None:
        # Check if range is within safety range
        new_range = list(new_range)
        if len(new_range) == 0:
            self._current_valid_range = self.safety_range()
        elif len(new_range) == 2:
            safe_new_range = new_range.copy()
            safe_range = self.safety_range()
            if safe_new_range:
                if new_range[0] < safe_range[0]:
                    safe_new_range[0] = safe_range[0]
                    logger.info(
                        f"{self.name}: new current_valid_range"
                        " not within safety range. Taking lower "
                        "safety limits instead."
                    )
                if new_range[1] > safe_range[1]:
                    print(new_range[1])
                    print(safe_range[1])
                    safe_new_range[1] = safe_range[1]
                    logger.info(
                        f"{self.name}: new current_valid_range"
                        " not within safety range. Taking upper "
                        "safety limits instead."
                    )
            self._current_valid_range = safe_new_range
        else:
            logger.error(self.label() + ": Incorrect current valid range.")
            raise ValueError

    def compute_transition_voltage(self) -> float:
        """"""
        return self._transition_voltage

    def set_transition_voltage(self, new_T: float) -> None:

        if new_T >= self.safety_range()[0] and new_T <= self.safety_range()[1]:
            self._transition_voltage = new_T
        else:
            logger.error(
                f"Gate {self.label() }: " "Invalid transition voltage. Keeping old one."
            )
            raise ValueError
