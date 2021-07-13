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
    positive = 0
    negative = 1


class DeviceState(Enum):
    pinchedoff = 0
    opencurrent = 1
    singledot = 2
    doubledot = 3
    undefined = 4


@dataclass
class RangeChangeSetting:
    relative_range_change: float = 0.1
    max_range_change: float = 0.05
    min_range_change: float = 0.01


class DotTuner(Tuner):
    """
    """
    def __init__(
        self,
        name: str,
        data_settings: DataSettings,
        classifiers: Classifiers,
        setpoint_settings: SetpointSettings,
    ) -> None:
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
        """Does not reset any tuning"""

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
        """ """
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
        """ """
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
        """ """
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
        """ # Narrow down plunger ranges: """
        good_plunger_ranges = False
        while not good_plunger_ranges:
            barrier_changes = self.set_new_plunger_ranges(
                device,
                plunger_barrier_pairs=device_layout.plunger_barrier_pairs()
            )
            if barrier_changes is not None:
                    self.adjust_helper_and_outer_barriers(
                        device,
                        device_layout,
                        barrier_changes,
                    )
            else:
                good_plunger_ranges = True

    def adjust_helper_and_outer_barriers(
        self,
        device,
        device_layout: Type[DeviceLayout],
        barrier_changes: Dict[int, VoltageChangeDirection],
    ):
        new_v_change_dir = self.update_voltages_based_on_directives(
            device, barrier_changes
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
    ):
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
                device, gate_ids=device_layout.outer_barriers()
            )

    def set_helper_gate(
        self,
        device: Device,
        helper_gate_id: int = DoubleDotLayout.helper_gate(),
        gates_for_init_range_measurement: Sequence[int] = DoubleDotLayout.barriers(),
    ) -> None:
        """ top_barrier in 2D doubledot device"""
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
            voltage_ranges = device.initial_valid_ranges()[helper_gate_id]

        device.current_valid_ranges({helper_gate_id: voltage_ranges})
        device.gates[helper_gate_id].voltage(voltage_ranges[1])

    def update_gate_configuration(
        self,
        device: Device,
        last_result: TuningResult,
        target_state: DeviceState,
        helper_gate_id: int = 0,
        central_barrier_id: int = 3,
        outer_barrier_ids: Sequence[int] = (1, 4),
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
    ) -> None:
        """
        Choose new gate voltages when previous tuning did not result in any
        good regime.
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
        central_barrier_id: int = 3,
    ):
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
        barrier_gate_ids: Sequence[int] = (1, 4)
    ) -> Dict[int, VoltageChangeDirection]:
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
                for gate_id in barrier_gate_ids:
                    barrier_directives[gate_id] = v_dir
        return barrier_directives

    def adjust_all_barriers_loop(
        self,
        device,
        target_state: DeviceState,
        initial_voltage_update: Dict[int, VoltageChangeDirection],
        helper_gate_id: int = 0,
        central_barrier_id: int = 3,
        outer_barriers_id: Sequence[int] = (1, 4),
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
    ):
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
        helper_gate_id: int = 0,
        central_barrier_id: int = 3,
        outer_barriers_id: Sequence[int] = (1, 4),
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
    ) -> Optional[VoltageChangeDirection]:
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
        )
        return new_direction

    def set_outer_barriers(
        self,
        device: Device,
        gate_ids: Sequence[float] = (1, 4),
        tolerance: float = 0.1,
    ) -> Optional[VoltageChangeDirection]:
        """
        Will not update upper valid range limit. We assume it has been
        determined in the beginning with central_barrier = 0 V and does
        not change.
        new_voltage = T + 2 / 3 * abs(T - H)
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
        """"""
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
        tolerance: float = 0.1,
    ) -> Optional[VoltageChangeDirection]:
        """
        returns new direction if touching limits
        In that case some other gate needs to be set more negative.
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
            new_direction = check_new_voltage(new_voltage, gate, tolerance)
            if new_direction is None:
                device.gates[gate_id].voltage(new_voltage)
                logger.info(
                    f"Choosing new voltage for {gate}: {new_voltage}"
                )

        return new_direction

    def set_new_plunger_ranges(
        self,
        device: Device,
        plunger_barrier_pairs: List[Tuple[int, int]] = DoubleDotLayout.plunger_barrier_pairs(),
    ) -> Optional[Dict[int, VoltageChangeDirection]]:
        """
        noise_floor and dot_signal_threshold compared to normalized signal
        checks if barriers need to be adjusted, depending on min and max signal
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
        """ """
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
):
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
