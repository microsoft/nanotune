# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from dataclasses import asdict, dataclass, field
import logging
from contextlib import contextmanager
from typing import Generator, Iterable, List, Optional, Sequence, Tuple, Dict, Union

import numpy as np
import qcodes as qc
from nanotune.device.device import Device, NormalizationConstants
from nanotune.device.device_channel import DeviceChannel
from nanotune.device_tuner.tuningresult import MeasurementHistory, TuningResult
from nanotune.tuningstages.gatecharacterization1d import GateCharacterization1D
from nanotune.tuningstages.settings import (DataSettings, SetpointSettings,
    Classifiers)
from nanotune.tuningstages.chargediagram import ChargeDiagram
logger = logging.getLogger(__name__)


@contextmanager
def set_back_voltages(gates: List[DeviceChannel]) -> Generator[None, None, None]:
    """Context manager setting back gate voltages to their initial values. If
    gates need to be set in a specific
    order, then this order needs to be respected on the list 'gates'.

    Args:
        gates: List of DeviceChannels, i.e. gates of a Device.

    Returns:
        generator returning None
    """
    initial_voltages = []
    for gate in gates:
        initial_voltages.append(gate.voltage())
    try:
        yield
    finally:
        for ig, gate in enumerate(gates):
            gate.voltage(initial_voltages[ig])


@dataclass
class TuningHistory:
    """Container holding tunign results of several devices.

    Attributes:
        results: Mapping device name to an instance of
            MeasurementHistory.
    """
    results: Dict[str, MeasurementHistory] = field(default_factory=dict)

    def update(
        self,
        device_name: str,
        new_result: Union[TuningResult, MeasurementHistory],
    ):
        """Adds a tuning result - either an instance of TuningResult or
        MeasurementHistory.

        Args:
            device_name: name of Device instance.
            new_result: Either instance of
                TuningResult or MeasurementHistory.
        """
        if device_name not in self.results.keys():
            self.results[device_name] = MeasurementHistory(device_name)
        if isinstance(new_result, MeasurementHistory):
            self.results[device_name].update(new_result)
        elif isinstance(new_result, TuningResult):
            self.results[device_name].add_result(new_result)
        else:
            raise NotImplementedError(
                f"Can not add result of type {type(new_result)}")


class Tuner(qc.Instrument):
    """Tuner base class. It implements common methods used in both device
    characterization and dot tuning.

    Attributes:
        classifiers: a setting.Classifiers instance
            holding all required classifiers. Eg. pinchoff.
        data_settings: A settings.DataSettings instance with
            data related information such as `db_name` and
            `normalization_constants'.
        setpoint_settings: A settings.SetpointSettings
            instance with setpoint related information such as
            `voltage_precision`.
        tuning_history: A TuningHistory instance holding all
            tuning results.
    """

    def __init__(
        self,
        name: str,
        data_settings: DataSettings,
        classifiers: Classifiers,
        setpoint_settings: SetpointSettings,
    ) -> None:
        """Tuner init.

        Args:
            classifiers: a setting.Classifiers instance
                holding all required classifiers. Eg. pinchoff.
            data_settings: A settings.DataSettings instance with
                data related information such as `db_name` and
                `normalization_constants'.
            setpoint_settings: A settings.SetpointSettings
                instance with setpoint related information such as
                `voltage_precision`.
        """
        super().__init__(name)

        self.classifiers = classifiers

        self.data_settings = data_settings
        self.setpoint_settings = setpoint_settings
        self.tuning_history: TuningHistory = TuningHistory()

    @property
    def setpoint_settings(self) -> SetpointSettings:
        """Setpoint settings property."""
        return self._setpoint_settings

    @setpoint_settings.setter
    def setpoint_settings(self, new_settings: SetpointSettings) -> None:
        """Setpoint settings property setter. Overwrites all previous settings
        and adds new ones to static metadata.

        Args:
            new_settings (SetpointSettings): an SetpointSettings instance with
                new settings.
        """
        self._setpoint_settings = new_settings
        self.metadata.update({'setpoint_settings': asdict(new_settings)})

    @property
    def data_settings(self) -> DataSettings:
        """Data settings property."""
        return self._data_settings

    @data_settings.setter
    def data_settings(self, new_settings: DataSettings) -> None:
        """Data settings property setter. Overwrites all previous settings
        and adds new ones to static metadata.

        Args:
            new_settings: an DataSettings instance with
                new settings.
        """
        self._data_settings = new_settings
        self.metadata.update({'data_settings': asdict(new_settings)})

    def update_normalization_constants(self, device: Device):
        """Measures and sets normalization constants of a given device
        It gets the maximum and minimum signal for all
        readout methods in device.readout. It will first
        set all gate voltages to their most negative allowed values to record
        the minimal signal and then to their most positive allowed ones,
        measuring the maximum signal.
        The device is put back into state where it was before, with gates being
        set in the order they are listed in device.gates.

        Args:
            device: the device to measure, an instance of nt.Device
        """
        available_readout = device.readout.available_readout()

        normalization_constants = NormalizationConstants()

        with set_back_voltages(device.gates):
            device.all_gates_to_lowest()
            for read_meth in available_readout.keys():
                val = getattr(device.readout, read_meth).get()
                prev = getattr(normalization_constants, read_meth)
                new_tuple = (val, prev[1])
                setattr(normalization_constants, read_meth, new_tuple)

            device.all_gates_to_highest()
            for read_meth in available_readout.keys():
                val = getattr(device.readout, read_meth).get()
                prev = getattr(normalization_constants, read_meth)
                new_tuple = (prev[0], val)
                setattr(normalization_constants, read_meth, new_tuple)
        device.normalization_constants = normalization_constants

    def characterize_gate(
        self,
        device: Device,
        gate: DeviceChannel,
        use_safety_voltage_ranges: bool = False,
        iterate: bool = False,
        voltage_precision: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> TuningResult:
        """Characterizes a single DeviceChannel/gate of a device. Other than
        the gate swept, it does not set any voltages.

        Args:
            device: device to tune.
            gate: DeviceChannel instance/the gate to
                characterize.
            use_safety_voltage_ranges: whether or not the entire safe
                voltage range should be swept. If `False`, the gate's
                `current_valid_range` will be used.
            iterate: whether the gate should be characterized again over
                an extended voltage if a poor result is measured. Only makes
                sense if `use_safety_voltage_ranges=False`. The new ranges are
                determined based on the range update directives returned by
                PinchoffFit.
            voltage_precision: optional voltage precision, i.e. the
                voltage difference between subsequent setpoints. If none given,
                the value in self.data_settings is taken. The optional input
                here can be used to temporarily overwrite the default value.
            comment: optional string added to the tuning result.

        Return:
            TuningResult
        """
        if comment is None:
            comment = f"Characterizing {gate}."
        if not self.classifiers.is_pinchoff_classifier():
            raise KeyError("No pinchoff classifier found.")

        if use_safety_voltage_ranges:
            v_range = gate.safety_voltage_range()
            iterate = False
        else:
            v_range = device.current_valid_ranges()[gate.gate_id]

        setpoint_settings = self.measurement_setpoint_settings(
            [gate.voltage], [v_range], [gate.safety_voltage_range()],
            voltage_precision,
        )
        stage = GateCharacterization1D(
            data_settings=self.measurement_data_settings(device),
            setpoint_settings=setpoint_settings,
            readout=device.readout,
            classifier=self.classifiers.pinchoff,  # type: ignore
        )
        tuningresult = stage.run_stage(iterate=iterate)
        tuningresult.status = device.get_gate_status()
        tuningresult.comment = comment

        self.tuning_history.update(device.name, tuningresult)

        return tuningresult

    def measure_initial_ranges_2D(
        self,
        device: Device,
        gate_to_set: DeviceChannel,
        gates_to_sweep: Sequence[DeviceChannel],
        voltage_step: float = 0.2,
    ) -> Tuple[Tuple[float, float], MeasurementHistory]:
        """Estimates the initial valid voltage range of capacitively coupled
        gates - the range in which `gate_to_set` in combination with
        `gates_to_sweep` is able to deplete the electron gas nearby.
        Example: a 2DEG device with a top barrier, for which we wish to
        determine the range within which all relevant gates, here left, central
        and right barrier, pinch off. It can also be used to determine the
        range of a bottom gate beneath a 1D system.
        It executes `get_pairwise_pinchoff` twice: First it determines the least
        negative voltage of `gate_to_set` for which all `gates_to_sweep` pinch
        off by decreasing `gate_to_set` by `voltage_step` and characterizing
        all `gates_to_sweep`. It retains the gate which pinched off last and
        gets a pairwise pinchoff by setting this last gate to
        descresing voltages, characterizing `gate_to_set` at each value. The
        first time `gate_to_set` pinches off the `low_voltage`, or `L`, is
        retained as the upper range limit.

        Args:
            device: device to tune.
            gate_to_set: DeviceChannel instance of the gate
                for which an initial valid range should be determined.
            gates_to_sweep: DeviceChannel instances of the gates which
                couple to `gate_to_set` and thus affect its valid range.
            voltage_step: Voltage difference between consecutive gate
                voltage sets, i.e. by how much e.g. `gate_to_set` will be
                decreased at each iteration.

        Returns:
            Tuple[float, float]: Valid range of `gate_to_set`.
            MeasurementHistory: Collection of tuning results.
        """
        if self.classifiers.pinchoff is None:
            raise KeyError("No pinchoff classifier.")
        device.all_gates_to_highest()

        v_steps = linear_voltage_steps(
            gate_to_set.safety_voltage_range(), voltage_step)

        (measurement_result,
         last_gate_to_pinchoff,
         last_voltage) = self.get_pairwise_pinchoff(
            device,
            gate_to_set,
            gates_to_sweep,
            v_steps,
        )
        min_voltage = last_voltage

        device.all_gates_to_highest()

        # Swap top_barrier and last barrier to pinch off agains it to
        # determine opposite corner of valid voltage space.
        v_steps = linear_voltage_steps(
            last_gate_to_pinchoff.safety_voltage_range(), voltage_step)

        (measurement_result2,
         _,
         _) = self.get_pairwise_pinchoff(
            device,
            last_gate_to_pinchoff,
            [gate_to_set],
            v_steps,
        )
        # The line below is correct with max_voltage = ... "low_voltage"
        features = measurement_result2.last_added.ml_result["features"]
        max_voltage = features["transport"]["low_voltage"]

        measurement_result.update(measurement_result2)
        device.all_gates_to_highest()

        return (min_voltage, max_voltage), measurement_result

    def get_pairwise_pinchoff(
        self,
        device: Device,
        gate_to_set: DeviceChannel,
        gates_to_sweep: Sequence[DeviceChannel],
        voltages_to_set: Iterable[float],
    ) -> Tuple[MeasurementHistory, DeviceChannel, float]:
        """Determines voltage of `gate_to_set` at which all `gates_to_sweep`
        pinch off. Decreased `gate_to_set`'s voltage and characterizes those
        `gates_to_sweep` which have not pinched off at previous iterations.

        Args:
            device: device to tune.
            gate_to_set: DeviceChannel instance of the gate
                for which an initial valid range should be determined.
            gates_to_sweep: DeviceChannel instances of the gates which
                couple to `gate_to_set` and thus affect its valid range.
            voltages_to_set: sequence of voltages which `gate_to_set` is set
                and for which `gates_to_sweep` are characterized.

        Returns:
            MeasurementHistory:
            DeviceChannel: Gate to pinch off last (as voltage of `gate_to_set`
                is decreased).
            float: voltage `gate_to_set` at which all `gates_to_sweep` pinched
                off.
        """
        measurement_result = MeasurementHistory(device.name)
        layout_ids = [g.gate_id for g in gates_to_sweep]
        skip_gates = dict.fromkeys(layout_ids, False)

        for last_voltage in voltages_to_set:
            gate_to_set.voltage(last_voltage)

            for gate in gates_to_sweep:
                if not skip_gates[gate.gate_id]:
                    sub_tuning_result = self.characterize_gate(
                        device,
                        gate,
                        use_safety_voltage_ranges=True,
                        comment=f"Measuring initial range of {gate.full_name} \
                            with {gate_to_set.full_name} at {last_voltage}."
                    )
                    measurement_result.add_result(sub_tuning_result)
                    if sub_tuning_result.success:
                        skip_gates[gate.gate_id] = True
                        last_gate_to_pinchoff = gate

            if all(skip_gates.values()):
                break

        return measurement_result, last_gate_to_pinchoff, last_voltage

    def measurement_data_settings(
        self,
        device: Device,
    ) -> DataSettings:
        """Returns data settings with device specific information, e.g.
        normalization constants added.

        Args:
            device: device to tune.

        Returns:
            DataSettings: new data settings including normalization constants.
        """
        new_datasettings = DataSettings(**asdict(self.data_settings))
        norm_csts = device.normalization_constants
        new_datasettings.normalization_constants = norm_csts
        return new_datasettings

    def measurement_setpoint_settings(
        self,
        parameters_to_sweep: Sequence[qc.Parameter],
        ranges_to_sweep: Sequence[Sequence[float]],
        safety_voltage_ranges: Sequence[Sequence[float]],
        voltage_precision: Optional[float] = None,
    ) -> SetpointSettings:
        """Returns setpoint settings with updated `parameters_to_sweep`,
        `ranges_to_sweep`, `safety_voltage_ranges` and `voltage_precision`.

        Args:
            parameters_to_sweep: List of QCoDeS
                parameters which will be swept.
            ranges_to_sweep: List of voltage
                ranges, in the same order as `parameters_to_sweep`.
            safety_voltage_ranges: List of safe
                voltage ranges, in the same order as `parameters_to_sweep`.
            voltage_precision: optional float, voltage difference between
                setpoints.

        Returns:
            SetpointSettings: updated setpoint settings.
        """
        new_settings = SetpointSettings(**asdict(self.setpoint_settings))
        new_settings.parameters_to_sweep = parameters_to_sweep
        new_settings.ranges_to_sweep = ranges_to_sweep
        new_settings.safety_voltage_ranges = safety_voltage_ranges
        if voltage_precision is not None:
            new_settings.voltage_precision = voltage_precision
        return new_settings


    def get_charge_diagram(
        self,
        device: Device,
        gates_to_sweep: Sequence[DeviceChannel],
        use_safety_voltage_ranges: bool = False,
        iterate: bool = False,
        voltage_precision: Optional[float] = None, #0.02,
        comment: Optional[str] = None,
    ) -> TuningResult:
        """Measures a charge diagram by sweeping `gates_to_sweep`. The returned
        TuningResult instance also contains information about dot segments,
        such as their voltage sub-ranges and classification outcome
        (`tuningresult.ml_result['dot_segments']`).
        The tuning results is added to tuner's `tuning_history`.

        Args:
            device: device to tune.
            gates_to_sweep: DeviceChannel instances
                of gates to sweep.
            use_safety_voltage_ranges: whether entire safety voltage
                range should be swept. Current valid range is taken if not.
                Default is False.
            iterate: whether subsequent diagrams with over extended
                ranges should be taken if the measurement is classified as poor/
                not the desired regime.
            voltage_precision: optional float, voltage difference between
                setpoints. If none given,
                the value in self.data_settings is taken. The optional input
                here can be used to temporarily overwrite the default value.
            comment: optional comment added to the tuning result.

        Returns:
            TuningResult: tuning result including classification result and
                dot segment info (`tuningresult.ml_result['dot_segments']`).
        """
        if comment is None:
            comment = f"Taking charge diagram of {gates_to_sweep}."
        if not self.classifiers.is_dot_classifier():
            raise ValueError(f"Not the right classifiers found for dots.")

        if use_safety_voltage_ranges:
            v_ranges = [g.safety_voltage_range() for g in gates_to_sweep]
            iterate = False
        else:
            valid_ranges = device.current_valid_ranges()
            v_ranges = [valid_ranges[g.gate_id] for g in gates_to_sweep]

        setpoint_settings = self.measurement_setpoint_settings(
                [g.voltage for g in gates_to_sweep],
                v_ranges,
                [g.safety_voltage_range() for g in gates_to_sweep],
                voltage_precision,
            )
        stage = ChargeDiagram(
            data_settings=self.measurement_data_settings(device),
            setpoint_settings=setpoint_settings,
            readout=device.readout,
            classifiers=self.classifiers,
        )

        tuningresult = stage.run_stage(iterate=iterate)
        tuningresult.status = device.get_gate_status()
        tuningresult.comment = comment

        self.tuning_history.update(device.name, tuningresult)

        logger.info(
            f"ChargeDiagram stage finished: " \
            f"success: {tuningresult.success}, " \
            f"termination_reason: {tuningresult.termination_reasons}"
        )
        return tuningresult


def linear_voltage_steps(
    voltage_range: Sequence[float],
    voltage_step: float,
) -> Sequence[float]:
    """Returns linearly spaced setpoints.

    Args:
        voltage_range: interval within which setpoints
            should be computed.
        voltage_step: voltage difference between setpoints.

    Returns:
        Sequence[float]: linearly spaced setpoints.
    """
    n_steps = int(abs(voltage_range[0] - voltage_range[1]) / voltage_step)
    return np.linspace(voltage_range[1], voltage_range[0], n_steps)
