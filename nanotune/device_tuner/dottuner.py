from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Any, Type

import numpy as np

import nanotune as nt
from nanotune.device.device import Device
from nanotune.device.device_channel import DeviceChannel
from nanotune.device.device_layout import DeviceLayout, DoubleDotLayout
from nanotune.device_tuner.tuner import (Tuner, DataSettings, SetpointSettings,
    Classifiers)
from nanotune.device_tuner.tuningresult import MeasurementHistory, TuningResult

logger = logging.getLogger(__name__)

from enum import Enum
class VoltageChangeDirection(Enum):
    """Direction in which gate voltages can be changes.

    Attributes:
        positive (0): increase voltage.
        negative (1): decrease voltage.
    """
    positive = 0
    negative = 1


class DeviceState(Enum):
    """Possible device states.

    Attributes:
        pinchedoff (0): pinched off, no measurable current through the device.
        opencurrent (1): open regime, high current through the device (possibly
            no or few gates set to small voltages.)
        singledot (2): a single quantum dot is formed.
        doubledot (3): two dots next to each other are formed.
        undefined (4): either unknown regime, dots about to form or state at
            initialization.
    """
    pinchedoff = 0
    opencurrent = 1
    singledot = 2
    doubledot = 3
    undefined = 4


@dataclass
class RangeChangeSetting:
    """Settings defining how voltage ranges should be updated when needed.

    Attributes:
        relative_range_change (float): percentage of previous range of how much
            the new range should differ/be moved towards either positive or
            negative values.
        max_range_change (float): absolute maximum range change in V, i.e. max
            of how much the range will change at once.
        min_range_change (float): absolute minimum range change in V, i.e. min
            of how much the range will change at once.
    """
    relative_range_change: float = 0.1
    max_range_change: float = 0.05
    min_range_change: float = 0.01
    tolerance: float = 0.1


class DotTuner(Tuner):
    """`Tuner` sub-class with dot tuning procedures.

    Attributes:
        classifiers (Classifiers): a setting.Classifiers instance
            holding all required classifiers. Eg. pinchoff.
        data_settings (DataSettings): A settings.DataSettings instance with
            data related information such as `db_name` and
            `normalization_constants'.
        setpoint_settings (SetpointSettings): A settings.SetpointSettings
            instance with setpoint related information such as
            `voltage_precision`.
        tuning_history (TuningHistory): A TuningHistory instance holding all
            tuning results.
    """
    def __init__(
        self,
        name: str,
        data_settings: DataSettings,
        classifiers: Classifiers,
        setpoint_settings: SetpointSettings,
    ) -> None:
        """DotTuner init. Calling Tuner init function without any other
        instantiation.

        Args:
            classifiers (Classifiers): a setting.Classifiers instance
                holding all required classifiers. Eg. pinchoff.
            data_settings (DataSettings): A settings.DataSettings instance with
                data related information such as `db_name` and
                `normalization_constants'.
            setpoint_settings (SetpointSettings): A settings.SetpointSettings
                instance with setpoint related information such as
                `voltage_precision`.
        """
        super().__init__(
            name,
            data_settings,
            classifiers,
            setpoint_settings,
        )

    def tune(
        self,
        device: Device,
        device_layout: Type[DeviceLayout] = DoubleDotLayout,
        target_state: DeviceState = DeviceState.doubledot,
        max_iter: int = 15,
        take_high_res: bool = False,
        continue_tuning: bool = False,
    ) -> Tuple[bool, MeasurementHistory]:
        """

        Args:
            device (nt.Device): device to tune.
        """

        if not self.classifiers.is_dot_classifier():
            raise ValueError("Not all dot classifiers found.")

        device.all_gates_to_highest()
        self.update_normalization_constants(device)

        self.set_helper_gate(
            device,
            helper_gate_id=device_layout.helper_gate(),
            gates_for_init_range_measurement=device_layout.barriers(),
        )

        success = self.tune_dot_regime(
            device=device,
            device_layout=device_layout,
            target_state=target_state,
            max_iter=max_iter,
            take_high_res=take_high_res,
            continue_tuning=continue_tuning,
        )

        return success, self.tuning_history.results[device.name]

    def tune_dot_regime(
        self,
        device: Device,
        device_layout: Type[DeviceLayout] = DoubleDotLayout,
        target_state: DeviceState = DeviceState.doubledot,
        max_iter: int = 15,
        take_high_res: bool = False,
        continue_tuning: bool = False,
    ) -> bool:
        """

        Args:
            device (nt.Device): device to tune.

        """
        self.set_central_and_outer_barriers(
            device, device_layout, target_state
        )
        done = False
        n_iter = 0
        success = False
        while not done and n_iter <= max_iter:
            n_iter += 1
            self.set_valid_plunger_ranges(device, device_layout)
            tuningresult = self.get_charge_diagram(
                device,
                [device.gates[gid] for gid in device_layout.plungers()],
                iterate=True,
            )
            if tuningresult.success and take_high_res:
                self.take_high_resolution_diagram(
                    device,
                    device_layout.plungers(),
                    take_segments=True,
                )
            done = tuningresult.success
            success = tuningresult.success
            if continue_tuning:
                done = False
                logger.warning("Continue tuning regardless of the outcome.")

            if not done:
                self.update_gate_configuration(
                    device, tuningresult, target_state,
                )

        if n_iter >= max_iter:
            logger.info(
                f"Tuning {device.name}: Max number of iterations" \
                f"reached.")
        return success

    def take_high_resolution_diagram(
        self,
        device,
        gate_ids: List[int] = DoubleDotLayout.plungers(),
        target_state: DeviceState = DeviceState.doubledot,
        take_segments: bool = True,
        voltage_precisions: Tuple[float, float] = (0.0005, 0.0001),
    ):
        """

        Args:
            device (nt.Device): device to tune.

        """
        logger.info("Take high resolution of entire charge diagram.")
        tuningresult = self.get_charge_diagram(
            device,
            [device.gates[gid] for gid in gate_ids],
            voltage_precision=voltage_precisions[0],
        )
        if take_segments:
            self.take_high_res_dot_segments(
                device,
                dot_segments=tuningresult.ml_result['dot_segments'],
                gate_ids=gate_ids,
                target_state=target_state,
                voltage_precision=voltage_precisions[1],
            )

    def take_high_res_dot_segments(
        self,
        device,
        dot_segments: Dict[int, Any],
        gate_ids: List[int] = DoubleDotLayout.plungers(),
        target_state: DeviceState = DeviceState.doubledot,
        voltage_precision: float= 0.0001,
    ):
        """

        Args:
            device (nt.Device): device to tune.

        """
        logger.info("Take high resolution of good charge diagram segments.")

        for r_id in dot_segments.keys():
            if dot_segments[r_id]["predicted_regime"] == target_state.value:
                v_rgs = dot_segments[r_id]['voltage_ranges']
                for g_id, v_range in zip(gate_ids, v_rgs):
                    device.current_valid_ranges({g_id: v_range})
                tuningresult = self.get_charge_diagram(
                    device,
                    [device.gates[gid] for gid in gate_ids],
                    voltage_precision=voltage_precision,
                )

    def set_valid_plunger_ranges(
        self,
        device,
        device_layout: Type[DeviceLayout],
    ):
        """

        Args:
            device (nt.Device): device to tune.

        """
        good_plunger_ranges = False
        while not good_plunger_ranges:
            barrier_changes = self.set_new_plunger_ranges(
                device,
                plunger_barrier_pairs=device_layout.plunger_barrier_pairs()
            )
            if barrier_changes is not None:
                    self.adjust_outer_barriers_possibly_helper_gate(
                        device,
                        device_layout,
                        barrier_changes,
                    )
            else:
                good_plunger_ranges = True

    def adjust_outer_barriers_possibly_helper_gate(
        self,
        device,
        device_layout: Type[DeviceLayout],
        barrier_changes: Dict[int, VoltageChangeDirection],
    ) -> None:
        """Adjusts outer barriers and if needed the helper gate. It first
        applies voltages changes specified by `barrier_changes` and if these
        are not successful, the helper gate is adjusted. This step is repeated
        until all barriers are set with a within their safety
        ranges.

        Args:
            device (nt.Device): device to tune.
            device_layout (DeviceLayout):
            barrier_changes (Dict[int, VoltageChangeDirection]):
        """
        new_v_change_dir = self.update_voltages_based_on_directives(
            device, barrier_changes,
        )

        if new_v_change_dir is not None:
            logger.info(
                "Outer barrier reached safety limit. " \
                "Setting new top barrier.\n"
            )
            _ = self.update_voltages_based_on_directives(
                device,
                {device_layout.helper_gate(): new_v_change_dir},
            )
            new_v_change_dir = self.set_outer_barriers(
                device, gate_ids=device_layout.outer_barriers(),
            )

            while new_v_change_dir is not None:
                _ = self.update_voltages_based_on_directives(
                    device,
                    {device_layout.helper_gate(): new_v_change_dir},
                )
                new_v_change_dir = self.set_outer_barriers(
                    device, gate_ids=device_layout.outer_barriers(),
                )

    def set_central_and_outer_barriers(
        self,
        device: Device,
        device_layout: Type[DeviceLayout] = DoubleDotLayout,
        target_state: DeviceState = DeviceState.doubledot,
    ) -> None:
        """Sets central and outer barriers given a target state. The central
        barrier is set first, to higher voltages for single than for the double
        dot regime, before setting outer barriers. If the latter is not
        successful, a loop adjusting the helper gate  as well as outer barriers
        is executed until all gates are happy.

        Args:
            device (nt.Device): device to tune.
            device_layout (DeviceLayout): device layout, default is
                DoubleDotLayout.
            target_state (DeviceState): target regime. Default is
                DeviceState.doubledot.
        """
        self.set_central_barrier(
            device,
            target_state=target_state,
            gate_id=device_layout.central_barrier(),
        )
        new_vltg_change_directions = self.set_outer_barriers(
            device, gate_ids=device_layout.outer_barriers(),
        )

        while new_vltg_change_directions is not None:
            _ = self.update_voltages_based_on_directives(
                device,
                {device_layout.helper_gate(): new_vltg_change_directions},
            )
            new_vltg_change_directions = self.set_outer_barriers(
                device, gate_ids=device_layout.outer_barriers(),
            )

    def set_helper_gate(
        self,
        device: Device,
        helper_gate_id: int = DoubleDotLayout.helper_gate(),
        gates_for_init_range_measurement: Sequence[int] = DoubleDotLayout.barriers(),
    ) -> None:
        """Sets the voltage of a device's helper gate, such as the top_barrier
        in a 2D doubledot device. Typically used in the very beginning of a
        tuning process and thus using `initial_valid_ranges`. If these are not
        set/the same as the gate's safety range, `measure_initial_ranges_2D`
        estimated a valid range.
        The gate's current valid range is set to either its initial valid
        range or the newly determined range. The voltage is set to the upper
        limit of the new current valid range.
        Raises an error if `helper_gate_id` is not in the device's gates list.

        Args:
            device (nt.Device): device to tune.
            helper_gate_id (int): gate ID of helper gate. Default is
                DoubleDotLayout.top_barrier().
            gates_for_init_range_measurement (Sequence[int]): list of gates
                coupled to the helper gate most, i.e which have the biggest
                influence on its valid range. Typically outer and central
                barriers.
        """
        if helper_gate_id  not in device._gates_dict.keys():
            raise KeyError("Gate not found.")

        init_ranges = device.initial_valid_ranges()[helper_gate_id]

        if init_ranges == device.gates[helper_gate_id].safety_voltage_range():
            gates_to_sweep = [
                device.gates[gate_id]
                for gate_id in gates_for_init_range_measurement
            ]
            (voltage_ranges,
             measurement_result) = self.measure_initial_ranges_2D(
                device,
                device.gates[helper_gate_id], gates_to_sweep,
            )
            self.tuning_history.update(device.name, measurement_result)
        else:
            voltage_ranges = init_ranges

        device.current_valid_ranges({helper_gate_id: voltage_ranges})
        device.gates[helper_gate_id].voltage(voltage_ranges[1])

    def update_gate_configuration(
        self,
        device: Device,
        last_result: TuningResult,
        target_state: DeviceState = DeviceState.doubledot,
        helper_gate_id: int = DoubleDotLayout.helper_gate(),
        central_barrier_id: int = DoubleDotLayout.central_barrier(),
        outer_barrier_ids: Sequence[int] = DoubleDotLayout.outer_barriers(),
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
    ) -> None:
        """Updates gate configuration based on last measurement result. If the
        previous measurement shows a good but not the target regime, the
        central barrier is adjusted. If the detected regime is poor, new outer
        barrier voltages are chosen. In both cases all barriers are adjusted
        using `adjust_all_barriers_loop` to ensure that the new settings sit
        well with all other barriers.
        Raises an error if no termination_reasons are found and the previous
        tuning result was poor.


        Args:
            device (nt.Device): device to tune.
            last_result (TuningResult): tuning result of last measurement/tuning
                stage.
            target_state (DeviceState): target regime. Default is
                DeviceState.doubledot. Raises an error for a state other than
                `pinchedoff`, `singledot` or `doubledot`.
            helper_gate_id (int): gate ID of helper gate. Default is
                DoubleDotLayout.top_barrier().
            central_barrier_id (int): gate ID of central barrier. Default is
                DoubleDotLayout.central_barrier()
            outer_barriers_id (Sequence[int]): gate IDs of outer barriers.
                Default is DoubleDotLayout.outer_barriers().
            range_change_setting (RangeChangeSetting): instance of
                RangeChangeSetting with relative, maximum and minimum range
                changes.
        """
        termination_reasons = last_result.termination_reasons
        if not termination_reasons and not last_result.success:
            raise ValueError("Unknown tuning outcome. Expect either a " \
                "successful tuning stage or termination reasons")

        if not termination_reasons and last_result.success:
            # A good regime was found, but not the right one.
            initial_update = self._update_dotregime_directive(
                target_state, central_barrier_id
            )
        elif termination_reasons and not last_result.success:
            # the charge diagram terminated unsuccessfully: no plunger ranges
            # were found to give a good diagram. The device might be too
            # pinched off or too open.
            initial_update = self._select_outer_barrier_directives(
                termination_reasons, outer_barrier_ids,
            )
        else:
            raise NotImplementedError('Unknown tuning status')

        self.adjust_all_barriers_loop(
            device,
            target_state,
            initial_update,
            helper_gate_id,
            central_barrier_id,
            outer_barrier_ids,
            range_change_setting=range_change_setting,
            )
        logger.info(
            f"Updated gate configuration to: {device.get_gate_status()}."
        )

    def _update_dotregime_directive(
        self,
        target_regime: DeviceState,
        central_barrier_id: int = DoubleDotLayout.central_barrier(),
    ) -> Dict[int, VoltageChangeDirection]:
        """Translates a target regime to voltage change directive for the
        central barrier. Used when a good, but not the desired regime was found.
        Ex: good single found but need double dot.

        Args:
            target_regime (DeviceState): target regime, either single or double
                dot. Raises an error otherwise.
            central_barrier_id (int): gate ID of central barrier. Default is
                DoubleDotLayout.central_barrier().

        Returns:
            Dict[int, VoltageChangeDirection]: new voltage change direction
                for the central barrier.
        """
        if target_regime == DeviceState.singledot:
            v_direction = VoltageChangeDirection.positive
        elif target_regime == DeviceState.doubledot:
            v_direction = VoltageChangeDirection.negative
        else:
            raise ValueError('Invalid target regime.')
        return {central_barrier_id: v_direction}

    def _select_outer_barrier_directives(
        self,
        termination_reasons: List[str],
        outer_barriers_id: Sequence[int] = DoubleDotLayout.outer_barriers(),
    ) -> Dict[int, VoltageChangeDirection]:
        """Translates termination reasons of a previous tuning stage into
        voltage change directives for outer barriers. Raises an error in case
        of invalid termination reasons.

        Args:
            termination_reasons (List[str]): reasons why a previous tuning stage
                finished.
            outer_barriers_id (Sequence[int]): gate IDs of outer barriers.
                Default is DoubleDotLayout.outer_barriers().

        Returns:
            Dict[int, VoltageChangeDirection]: voltage change directives for
                barriers. Mapping gate IDs onto `VoltageChangeDirection`s.
        """
        barrier_directives = {}
        for termin_reason in termination_reasons:
            v_dir = None
            if 'positive' in termin_reason:
                v_dir = VoltageChangeDirection.positive
            elif 'negative' in termin_reason:
                v_dir = VoltageChangeDirection.negative
            else:
                raise ValueError('No valid termination reason found.')
            if v_dir is not None:
                for gate_id in outer_barriers_id:
                    barrier_directives[gate_id] = v_dir
        return barrier_directives

    def adjust_all_barriers_loop(
        self,
        device,
        target_state: DeviceState,
        initial_voltage_update: Dict[int, VoltageChangeDirection],
        helper_gate_id: int = DoubleDotLayout.helper_gate(),
        central_barrier_id: int = DoubleDotLayout.central_barrier(),
        outer_barriers_id: Sequence[int] = DoubleDotLayout.outer_barriers(),
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
    ) -> None:
        """Adjusts all barriers in a loop until a satisfying (within safety
        limits) is found. It first updates voltages as specified in
        `initial_voltage_update` and then uses `adjust_all_barriers` until no
        new voltage update is required.

        Args:
            device (nt.Device): device to tune.
            target_state (DeviceState): target regime. Raises an error for a
                state other than
                `pinchedoff`, `singledot` or `doubledot`, although only
                `singledot` or `doubledot` make sense here.
            initial_voltage_update (Dict[int, VoltageChangeDirection]): dict
                mapping gate IDs onto direction in which their voltages need to
                be changed.
            helper_gate_id (int): gate ID of helper gate. Default is
                DoubleDotLayout.top_barrier().
            central_barrier_id (int): gate ID of central barrier. Default is
                DoubleDotLayout.central_barrier()
            outer_barriers_id (Sequence[int]): gate IDs of outer barriers.
                Default is DoubleDotLayout.outer_barriers().
            range_change_setting (RangeChangeSetting): instance of
                RangeChangeSetting with relative, maximum and minimum range
                changes.
        """
        new_v_direction = self.update_voltages_based_on_directives(
            device,
            initial_voltage_update,
            range_change_setting=range_change_setting,
        )
        while new_v_direction is not None:
            new_v_direction = self.adjust_all_barriers(
                device,
                target_state,
                new_v_direction,
                helper_gate_id,
                central_barrier_id,
                outer_barriers_id,
            )

    def adjust_all_barriers(
        self,
        device,
        target_state: DeviceState,
        voltage_change_direction: VoltageChangeDirection,
        helper_gate_id: int = DoubleDotLayout.helper_gate(),
        central_barrier_id: int = DoubleDotLayout.central_barrier(),
        outer_barriers_id: Sequence[int] = DoubleDotLayout.outer_barriers(),
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
    ) -> Optional[VoltageChangeDirection]:
        """Adjusts all barriers of a device based on a voltage change direction.
        Typically used when a device is too closed/pinched off or too open. It
        first updates the helper gate, then the central barrier and finally
        the outer barriers. If the outer barriers can not be updated, a
        voltage change direction is returned.

        Args:
            device (nt.Device): device to tune.
            target_state (DeviceState): target regime. Default is
                DeviceState.doubledot. Raises an error for a state other than
                `pinchedoff`, `singledot` or `doubledot`.
            voltage_change_direction (VoltageChangeDirection): general
                direction in which voltages need to change. Either
                `VoltageChangeDirection.positive` or
                `VoltageChangeDirection.negative`.
            helper_gate_id (int): gate ID of helper gate. Default is
                DoubleDotLayout.top_barrier().
            central_barrier_id (int): gate ID of central barrier. Default is
                DoubleDotLayout.central_barrier()
            outer_barriers_id (Sequence[int]): gate IDs of outer barriers.
                Default is DoubleDotLayout.outer_barriers().
            range_change_setting (RangeChangeSetting): instance of
                RangeChangeSetting with relative, maximum and minimum range
                changes.

        Returns:
            Optional[VoltageChangeDirection]: None is voltage updates have been
                successful, a new voltage change direction if not.
        """
        _ = self.update_voltages_based_on_directives(
            device,
            {helper_gate_id: voltage_change_direction},
            range_change_setting=range_change_setting,
        )
        self.set_central_barrier(
            device,
            target_state=target_state,
            gate_id=central_barrier_id,
        )
        new_direction = self.set_outer_barriers(
            device,
            gate_ids=outer_barriers_id,
            tolerance=range_change_setting.tolerance,
        )
        return new_direction

    def set_outer_barriers(
        self,
        device: Device,
        gate_ids: Sequence[int] = DoubleDotLayout.outer_barriers(),
        tolerance: float = 0.1,
    ) -> Optional[VoltageChangeDirection]:
        """Sets outer barriers. Each barrier is characterized and a new voltage
        chosen based on the extracted features `low_voltage` (L),
        `high_voltage` (H) and `transition_voltage` (T). The new voltage,
        `new_voltage = T + 2 / 3 * abs(T - H)` is checked and set if the check
        is passes (not too close to safety limits). The central barrier's
        current valid range is updated to `[L, H]` and transition voltage to
        `T`.

        Args:
            device (nt.Device): device to tune.
            gate_ids (Sequence[int]): barrier gate IDs, default is
                DoubleDotLayout.outer_barriers().
            tolerance (float): minimal voltage difference to keep to safety
                ranges. Default is 0.1.

        Returns:
            Optional[VoltageChangeDirection]: None is voltage updates have been
                successful, a new voltage change direction if not.
        """
        if isinstance(gate_ids, int):
            gate_ids = [gate_ids]
        main_readout_method = device.main_readout_method.name
        new_direction = None
        for barrier_id in gate_ids:
            barrier = device.gates[barrier_id]
            result = self.characterize_gate(
                device,
                barrier,
                comment="Characterize outer barriers before setting them.",
                use_safety_voltage_ranges=True,
            )
            features = result.ml_result["features"][main_readout_method]
            L, H = features["low_voltage"], features["high_voltage"]
            T = features["transition_voltage"]
            new_voltage = T + 2 / 3 * abs(T - H)

            new_direction = check_new_voltage(new_voltage, barrier, tolerance)
            if new_direction is None:
                barrier.voltage(new_voltage)
                device.current_valid_ranges({barrier_id: [L, H]})
                device.transition_voltages({barrier_id: T})
                logger.info(f"Set {barrier.name} to {new_voltage}.")

        return new_direction

    def set_central_barrier(
        self,
        device: Device,
        target_state: DeviceState = DeviceState.doubledot,
        gate_id: int = DoubleDotLayout.central_barrier(),
    ) -> None:
        """Sets the central barrier. The voltage chosen depends on the target
        state, with a more positive value when aiming for a single and
        a more negative one for a double dot. It characterizes the barrier
        chooses a value based on the determined `high_voltage` or signal
        strengths. For a double dot regime, the central barrier is set to the
        highest voltage at which the device show 2/3 of its open signal.

        Args:
            device (nt.Device): device to tune.
            target_state (DeviceState): target regime. Default is
                DeviceState.doubledot. Raises an error for a state other than
                `pinchedoff`, `singledot` or `doubledot`.
            gate_id (int): Default is DoubleDotLayout.central_barrier().
        """
        assert isinstance(target_state, DeviceState)
        barrier = device.gates[gate_id]
        result = self.characterize_gate(
            device,
            barrier,
            use_safety_voltage_ranges=True,
        )
        main_readout_method = device.main_readout_method.name
        features = result.ml_result["features"][main_readout_method]

        if target_state == DeviceState.singledot:
            barrier.voltage(features["high_voltage"])
        elif target_state == DeviceState.doubledot:
            data_id = result.data_ids[-1]
            ds = nt.Dataset(
                data_id,
                self.data_settings.db_name,
                db_folder=self.data_settings.db_folder,
            )
            signal = ds.data[main_readout_method].values
            voltage = ds.data[main_readout_method]["voltage_x"].values

            v_sat_idx = np.argwhere(signal < float(2 / 3))
            if len(v_sat_idx) > 0:
                v_index = v_sat_idx[-1][0]
            else:
                v_index = 0
            barrier.voltage(voltage[v_index])
        elif target_state == DeviceState.pinchedoff:
            barrier.voltage(features["low_voltage"])
        else:
            raise ValueError("Invalid DeviceState. Use singledot, doubledot " \
                "or pinchedoff.")

        new_range = [features["low_voltage"], features["high_voltage"]]
        device.current_valid_ranges({barrier.gate_id: new_range})
        logger.info(f"Central barrier set to {barrier.voltage()} with valid " \
            "range ({features['low_voltage']}, {features['high_voltage']}). ")

    def update_voltages_based_on_directives(
        self,
        device: Device,
        voltage_changes: Dict[int, VoltageChangeDirection],
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
    ) -> Optional[VoltageChangeDirection]:
        """Updates voltages based on `VoltageChangeDirection`s.
        Mainly used to update top or outer barriers. In case of outer barriers
        reaching limits, there is only one other gate (helper gate) to change.
        Returning only one direction is sufficient. In the current algorithm, it
        does not happen that one barrier requires the helper gate to be
        more positive and the other more negative.

        Args:
            device (nt.Device): device to tune.
            voltage_changes (Dict[int, VoltageChangeDirection]): dict mapping
                gate IDs onto a direction in which the voltage should be
                changed.
            range_change_setting (RangeChangeSetting): instance of
                RangeChangeSetting with relative, maximum and minimum range
                changes.

        Returns:
            Optional[VoltageChangeDirection]: None is voltage updates have been
                successful, a new voltage change direction if not.
        """
        new_direction = None
        for gate_id, direction in voltage_changes.items():
            gate = device.gates[gate_id]
            new_voltage = self.choose_new_gate_voltage(
                device,
                gate_id,
                direction,
                range_change_setting=range_change_setting,
            )
            new_direction = check_new_voltage(
                new_voltage, gate, range_change_setting.tolerance)
            if new_direction is None:
                device.gates[gate_id].voltage(new_voltage)
                logger.info(f"Choosing new voltage for {gate}: {new_voltage}")

        return new_direction

    def set_new_plunger_ranges(
        self,
        device: Device,
        plunger_barrier_pairs: List[Tuple[int, int]] = DoubleDotLayout.plunger_barrier_pairs(),
    ) -> Optional[Dict[int, VoltageChangeDirection]]:
        """Determines and sets new valid ranges for plungers.
        It characterizes each plunger individually using `characterize_plunger`,
        checking if a nearby barrier needs to adjusted.

        Args:
            device (nt.Device): device to tune.
            plunger_barrier_pairs (List[Tuple[int, int]]): list of tuples, where
                the first tuple item is plunger gate_id and the second a
                barrier gate_id, indicating which barrier needs to be changed
                if a plunger's range can not be updated.
                Ex: [(left_plunger.gate_id, left_barrier.gate_id)]

        Returns:
            Optional[Dict[int, VoltageChangeDirection]]: if required (plunger
                range updates were not successful), the required changes to be
                applied to barrier voltages are returned. If a dict is returned,
                it maps gate ID onto `VoltageChangeDirection`s. The new
                direction is None if new ranges have been set successfully.
        """
        if not isinstance(plunger_barrier_pairs, List):
            raise ValueError('Invalid plunger_barrier_pairs input.')

        barrier_changes_dict: Dict[int, VoltageChangeDirection] = {}

        for plunger_id, barrier_id in plunger_barrier_pairs:
            plunger = device.gates[plunger_id]
            new_range, device_state = self.characterize_plunger(
                device, plunger,
            )
            device.current_valid_ranges({plunger_id: new_range})

            if device_state == DeviceState.pinchedoff:
                drct = VoltageChangeDirection.positive
                barrier_changes_dict[barrier_id] = drct

            if device_state == DeviceState.opencurrent:
                drct = VoltageChangeDirection.negative
                barrier_changes_dict[barrier_id] = drct

        if not barrier_changes_dict:
            barrier_changes = None
        else:
            barrier_changes = barrier_changes_dict
        return barrier_changes

    def characterize_plunger(
        self,
        device: Device,
        plunger: DeviceChannel,
    ) -> Tuple[Tuple[float, float], DeviceState]:
        """Characterizes a plunger of a device by performing a gate
        characterization and checking whether the measured (normalized) signal
        is neither to high nor too low, i.e. whether the device is pinched off
        or in the open regime. A dot regime is found in-between these two
        regimes.

        Args:
            device (nt.Device): device to tune.
            plunger (DeviceChannel): DeviceChannel instance of a plunger.

        Returns:
            Tuple[float, float]:
            DeviceState: state of the device, `undefined` if the
                characterization was successful. If the measured signal is below
                `data_settings.noise_floor`, `DeviceState.pinchedoff` is
                returned. If the signal is too high, the value is
                `DeviceState.opencurrent`.
        """
        result = self.characterize_gate(
            device,
            plunger,
            use_safety_voltage_ranges=True,
            iterate=False,
        )
        readout = device.main_readout_method.name
        features = result.ml_result["features"][readout]
        new_range = (features["low_voltage"], features["high_voltage"])

        device_state = DeviceState.undefined
        if features["min_signal"] > self.data_settings.dot_signal_threshold:
            device_state = DeviceState.opencurrent
        if features["max_signal"] < self.data_settings.noise_floor:
            device_state = DeviceState.pinchedoff

        return new_range, device_state

    def choose_new_gate_voltage(
        self,
        device: Device,
        gate_id: int,
        voltage_change_direction: VoltageChangeDirection,
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
    ) -> float:
        """
        based on current_valid_range or safety_voltage_range if no
        current_valid_range is set

        Args:
            device (nt.Device): device to tune.
            gate_id (int): ID of gate to set.
            voltage_change_direction (VoltageChangeDirection): direction in
                which the voltage should be adjusted. Positive or negative.
            range_change_setting (RangeChangeSetting): instance of
                RangeChangeSetting with relative, maximum and minimum range
                changes.

        Returns:
            float: new gate voltage to set on gate with ID `gate_id`.
        """
        relative_range_change =  range_change_setting.relative_range_change
        max_range_change = range_change_setting.max_range_change
        min_range_change = range_change_setting.min_range_change
        assert isinstance(voltage_change_direction, VoltageChangeDirection)

        curr_v = device.gates[gate_id].voltage()
        sfty_rng = device.gates[gate_id].safety_voltage_range()
        L, H = device.current_valid_ranges()[gate_id]

        if voltage_change_direction == VoltageChangeDirection.negative:
            v_change = max(
                relative_range_change * abs(curr_v - L), min_range_change
            )
            v_change = min(max_range_change, v_change)
            new_v = curr_v - v_change
            if new_v < sfty_rng[0]:
                new_v = sfty_rng[0]

        if voltage_change_direction == VoltageChangeDirection.positive:
            v_change = max(
                relative_range_change * abs(curr_v - H), min_range_change
            )
            v_change = min(max_range_change, v_change)
            new_v = curr_v + v_change
            if new_v > sfty_rng[1]:
                new_v = sfty_rng[1]

        return new_v


def check_new_voltage(
    new_voltage: float,
    gate: DeviceChannel,
    tolerance: float=0.1,
) -> Optional[VoltageChangeDirection]:
    """Checks whether a voltage can be set on a gate. If not, a
    VoltageChangeDirection indicating how other gates need to be updated
    is returned.

    Args:
        new_voltage (float): voltage to check before it is set.
        gate (DeviceChannel): gate whose voltage should be set.
        tolerance (float): minimal voltage difference to keep to safety ranges.
            Default is 0.1.

    Returns:
        VoltageChangeDirection: Direction in which other gates should be
            changed if the new voltage can not be set.
    """
    safe_range = gate.safety_voltage_range()
    touching_limits = np.isclose(
        new_voltage, safe_range, atol=tolerance
    )
    new_direction = None
    if touching_limits[0]:
        new_direction = VoltageChangeDirection.negative
    if touching_limits[1]:
        new_direction = VoltageChangeDirection.positive
    return new_direction
