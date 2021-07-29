import copy
import logging
import time
import importlib
from contextlib import contextmanager
from math import isclose
from typing import List, Optional, Tuple, Union, Any, Sequence, Generator

import qcodes as qc
from qcodes import InstrumentChannel
from qcodes import validators as vals
from qcodes.station import Station

from nanotune.drivers.dac_interface import DACInterface, RelayState

logger = logging.getLogger(__name__)


class DeviceChannel(InstrumentChannel):
    """Class representing a channel of a device such as gate or ohmic.

    It emulates an existing intrument channel and adds convenience
    functionality for tuning. It implements its own voltage setting method
    which, for example, checks whether the value is within the safety range
    before starting to set anything - as opposed to qcodes. The emulated
    channel must be a channel of a subclass of
    `nanotune.drivers.dac_interface.DACInterface`. When setting a voltage,
    it can be either ramped or set, specified by the `use_ramp` parameter.
    A hardware ramp is performed if the underlying instrument channel
    supports it, which is indicated by the `supports_hardware_ramp`
    property. Otherwise QCoDeS' software ramp using delay and step
    properties is used.

    Parameters:
        gate_id: Integer identifier of the gate with respect to the device
            layout. Not to be set if it is an ohmic.
        ohmic_id: Integer identifier of the ohmic with respect to the
            device layout. Not to be set if it is a gate.
        label: String identifier to be printed on plots.
        safety_voltage_range: Range of voltages within which safe operation
            is guaranteed.
        use_ramp: Whether voltages should be ramped.
        ramp_rate: Rate at which voltages should be ramped. In V/s.
        max_voltage_step: Maximum voltage difference to be applied in a
            single step. Note that this step does not apply when `use_ramp`
            is set to False.
        post_delay: Delay to wait after a voltage is set, in seconds. If
            ramping in software (supports_hardware_ramp == False and
            use_ramp() == True and thus using QCoDeS' delay and step
            parameter) and `inter_delay` is not set, then `post_delay` will
            be used.
        inter_delay: Delay to wait between voltage setpoints, in seconds.
            If set to zero but `post_delay` is not zero, software ramp
            (supports_hardware_ramp == False and use_ramp() == True) will
            take the `post_delay` as delay between sets.
        supports_hardware_ramp: Bool indicating whether under the
            underlying instrument channel supports voltage ramping in
            hardware.
        relay_state: State of DAC relay, one of
            `nanotune.dac_interface.RelayState`'s members.
        frequency: Frequency parameter, part of the AWG functionality of
            the underlying channel (if supported - as defined by the
            instance of DACInterface instance).
        offset: Offset parameter, part of the AWG functionality of
            the underlying channel (if supported - as defined by the
            instance of DACInterface instance).
        amplitude: Amplitude parameter, part of the AWG functionality of
            the underlying channel (if supported - as defined by the
            instance of DACInterface instance).

    Methods:
        ground: Set relay_state to RelayState.ground
        float: Set relay_state to RelayState.floating
        set_voltage: Voltage setter.
    """
    def __init__(
        self,
        station: qc.Station,
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
        """DeviceChannel init.

        Args:
            channel: existing instrument channel, e.g. of a DAC: dac.ch01.
                Either a string or instrument channel instance.
            gate_id: Integer identifier of the gate with respect to the device
                layout. Not to be set if it is an ohmic.
            ohmic_id: Integer identifier of the ohmic with respect to the
                device layout. Not to be set if it is a gate.
            label: String identifier to be printed on plots.
            safety_voltage_range: Range of voltages within which safe operation
                is guaranteed.
            use_ramp: Whether voltages should be ramped.
            ramp_rate: Rate at which voltages should be ramped. In V/s.
            max_voltage_step: Maximum voltage difference to be applied in a
                single step. Note that this step does not apply when `use_ramp`
                is set to False.
            post_delay: Delay to wait after a voltage is set, in seconds. If
                ramping in software (supports_hardware_ramp == False and
                use_ramp() == True and thus using QCoDeS' delay and step
                parameter) and `inter_delay` is not set, then `post_delay` will
                be used.
            inter_delay: Delay to wait between voltage setpoints, in seconds.
                If set to zero but `post_delay` is not zero, software ramp
                (supports_hardware_ramp == False and use_ramp() == True) will
                take the `post_delay` as delay between sets.
        """
        self._channel = self._get_channel_instance(station, channel)

        if gate_id is not None and ohmic_id is not None:
            raise ValueError(
                f"Initializing DeviceChannel of {self._channel} as both gate \
                    and ohmic."
            )

        channel_name = self._channel.name
        if channel_name.startswith(self._channel.parent.name):
            channel_name = channel_name.replace(
                self._channel.parent.name+"_", "", 1
            )
        assert issubclass(self._channel.parent.__class__, DACInterface)

        static_metadata = kwargs.pop('metadata', {})
        static_metadata.update({
            'label': label,
            'gate_id':gate_id,
            'ohmic_id': ohmic_id}
        )

        if 'parent' not in kwargs.keys():
            kwargs['parent'] = self._channel.parent

        kwargs['name'] = channel_name
        super().__init__(
            metadata=static_metadata,
            **kwargs,
        )
        self.label = label
        self._gate_id = gate_id
        self._ohmic_id = ohmic_id

        super().add_parameter(
            name="voltage",
            label=f"{label} voltage",
            set_cmd=self.set_voltage,
            get_cmd=self._channel.get_voltage,
            vals=vals.Numbers(*safety_voltage_range),
        )

        self.inter_delay = inter_delay
        self.post_delay = post_delay
        self.max_voltage_step = max_voltage_step

        super().add_parameter(
            name="safety_voltage_range",
            label=f"{label} voltage safety range",
            set_cmd=self._set_safety_voltage_range,
            get_cmd=self._get_safety_voltage_range,
            initial_value=list(safety_voltage_range),
            vals=vals.Lists(),
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
            vals=vals.Enum(*[e for e in RelayState]),
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
    def gate_id(self) -> Optional[int]:
        """Device layout ID of a gate."""
        return self._gate_id

    @property
    def ohmic_id(self) -> Optional[int]:
        """Device layout ID of an ohmic."""
        return self._ohmic_id

    @property
    def supports_hardware_ramp(self) -> float:
        """Boolean indication whether the underlying instrument channel
        supports hardware ramp."""
        return self._channel.supports_hardware_ramp

    @property
    def post_delay(self) -> float:
        """Waiting time in seconds after a sequence of set operations."""
        return self._channel.get_voltage_post_delay()

    @post_delay.setter
    def post_delay(self, new_value: float) -> None:
        self._channel.set_voltage_post_delay(new_value)

    @property
    def inter_delay(self) -> float:
        """Waiting time in seconds between set operations."""
        return self._channel.get_voltage_inter_delay()

    @inter_delay.setter
    def inter_delay(self, new_value: float) -> None:
        self._channel.set_voltage_inter_delay(new_value)

    @property
    def max_voltage_step(self) -> float:
        """Maximum voltage difference between consecutive voltage sets. """
        return self._channel.get_voltage_step()

    @max_voltage_step.setter
    def max_voltage_step(self, new_value: float) -> None:
        self._channel.set_voltage_step(new_value)

    def ground(self) -> None:
        """Set relay state to ground."""
        self.relay_state(RelayState.ground)

    def float_relay(self) -> None:
        """Set relay state to float."""
        self.relay_state(RelayState.floating)

    def set_voltage(self, new_value: float, tol: float = 1e-5):
        """Voltage setter.

        It checks if the new value is within safety ranges
        before setting anything - as opposed to qcodes, which throws an error
        only once the safety range is reached.
        It ramps to the new value if ramping is enabled, using either software
        or hardware ramps - depending on `supports_hardware_ramp`. It's a
        blocking ramp.
        Args:
            new_value: New voltage value to set.
            tol: Tolerance in Volt. Used to check if ramp has finished.
        """
        safe_range = self.safety_voltage_range()
        if new_value < safe_range[0] - tol or new_value > safe_range[1] + tol:
            raise ValueError(
                f"Setting voltage outside of permitted range: \
                    {self.label} to {new_value}.")

        if self.supports_hardware_ramp and self.use_ramp():
            self._channel.ramp_voltage(new_value)
            while not isclose(new_value, self.voltage(), abs_tol=tol):
                time.sleep(0.01)
            time.sleep(self.post_delay)

        elif not self.supports_hardware_ramp and self.use_ramp():
            if self.inter_delay == 0 and self.post_delay > 0:
                delay = self.post_delay
            else:
                delay = self.inter_delay
            if self.max_voltage_step == 0 or delay == 0:
                logger.warning(
                    "Using software ramp without max_voltage_step \
                    or inter_delay set."
                )
            self._channel.voltage(new_value)

        elif not self.use_ramp():
            if abs(self.voltage() - new_value) > self.max_voltage_step:
                raise ValueError("Setting voltage in steps larger than \
                    max_voltage_step. Decrease \
                    max_voltage_step or set use_ramp(True).")
            with self._set_temp_inter_delay_and_step(0, 0):
                self._channel.voltage(new_value)

        else:
            raise ValueError(
                "Unknown voltage setting mode. Should I ramp or not?"
            )

    def _get_safety_voltage_range(self) -> List[float]:
        return copy.deepcopy(self._safety_voltage_range)

    def _set_safety_voltage_range(self, new_range: List[float]) -> None:
        """ """
        if len(new_range) != 2:
            raise ValueError("Invalid safety voltage range.")

        new_range = sorted(new_range)
        current_v = self.voltage()
        if current_v < new_range[0] or current_v > new_range[1]:
            raise ValueError('Current voltage not within new safety range.')

        self._safety_voltage_range = new_range
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

    def _get_channel_instance(
        self,
        station: Station,
        channel: Union[str, InstrumentChannel],
    ) -> InstrumentChannel:
        if isinstance(channel, str):
            def _parse_path(parent: Any, elem: Sequence[str]) -> Any:
                child = getattr(parent, elem[0])
                if len(elem) == 1:
                    return child
                return _parse_path(child, elem[1:])

            return _parse_path(station, channel.split("."))

        elif isinstance(channel, InstrumentChannel):
            instr_channel = channel
        else:
            raise ValueError(f'Invalid type for "channel" input.')
        return instr_channel

    @contextmanager
    def _set_temp_inter_delay_and_step(
        self,
        inter_delay: float,
        max_voltage_step: float,
    ) -> Generator[None, None, None]:
        """ """
        current_inter_delay = self.inter_delay
        current_step = self.max_voltage_step

        self.inter_delay = inter_delay
        self.max_voltage_step = max_voltage_step
        try:
            yield
        finally:
            self.inter_delay = current_inter_delay
            self.max_voltage_step = current_step
