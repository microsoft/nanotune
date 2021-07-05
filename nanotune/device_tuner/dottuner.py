import copy
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import nanotune as nt
from nanotune.device.device import Device, Readout, ReadoutMethods
from nanotune.device.device_channel import DeviceChannel
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


    def tune_2D(
        self,
        device: Device,
        target_state: str = "doubledot",
        max_iter: int = 100,
        take_high_res: bool = False,
    ) -> MeasurementHistory:
        """"""
        if not self.classifiers.is_dot_classifier():
            raise ValueError("Not all dot classifiers found.")

        device.all_gates_to_highest()
        self.update_normalization_constants(device)
        self.set_helper_gate(device)  # set top_barrier

        result = self.tune_1D(
            device,
            target_state=target_state,
            max_iter=max_iter,
            set_barriers=True,
            take_high_res=take_high_res,
        )
        return result

    def tune_1D(
        self,
        device: Device,
        target_state: str,
        max_iter: int = 100,
        set_barriers: bool = True,
        take_high_res: bool = False,
        reset_gates: bool = False,
        continue_tuning: bool = False,
        helper_gate_id: int = 0,
    ) -> Dict[Any, Any]:
        """Does not reset any tuning"""

        if reset_gates:
            device.all_gates_to_highest()

        done = False
        if set_barriers:
            self.set_central_barrier(device, target_state=target_state)
            (success,
             new_voltage_change_directions) = self.set_outer_barriers(device)
            while not success:
                _ = self.update_voltages_based_on_directives(
                    device, {helper_gate_id: new_voltage_change_directions},
                )
                # self.update_helper_gate(device, new_voltage_change_directions)
                (success,
                 new_voltage_change_directions) = self.set_outer_barriers(
                     device
                )

        plungers = [device.left_plunger, device.right_plunger]
        n_iter = 0
        while not done and n_iter <= max_iter:
            n_iter += 1
            good_barriers = False

            while not good_barriers:
                # Narrow down plunger ranges:
                (success,
                 barrier_changes) = self.set_new_plunger_ranges(
                    device
                )

                if not success:
                    out_br_success, new_v_change_dir = self.update_voltages_based_on_directives(
                        device, barrier_changes
                    )

                    if not out_br_success:
                        logger.info(
                            (
                                "Outer barrier reached safety limit. "
                                "Setting new top barrier.\n"
                            )
                        )
                        _ = self.update_voltages_based_on_directives(
                            device,
                            {helper_gate_id: new_v_change_dir},
                        )
                        (success,
                         new_v_change_dir) = self.set_outer_barriers(device)
                        while not success:
                            _ = self.update_voltages_based_on_directives(
                                device,
                                {helper_gate_id: new_v_change_dir},
                            )
                            (success,
                             new_v_change_dir) = self.set_outer_barriers(device)
                else:
                    good_barriers = True

            tuningresult = self.get_charge_diagram(
                device,
                plungers,
                signal_thresholds=[0.004, 0.1],
            )
            self.measurement_result_all = {device.name: tuningresult}
            logger.info(
                (
                    f"ChargeDiagram stage finished: {tuningresult.success}\n"
                    f"termination_reason: {tuningresult.termination_reasons}"
                )
            )
            segment_info = tuningresult.features["segment_info"]
            if target_state in segment_info[:, 2]:
                logger.info("Desired regime found.")
                done = True

            if done and take_high_res:

                logger.info("Take high resolution of entire charge diagram.")
                tuningresult = self.get_charge_diagram(
                    device,
                    plungers,
                    voltage_precision=0.0005,
                )
                self.measurement_result_all = {device.name: tuningresult}
                # self.measurement_result_all[device.name].add_result(
                #     tuningresult, f"chargediagram_highres_{n_iter}"
                # )
                logger.info("Take high resolution of charge diagram segments.")
                segment_info = tuningresult.features["segment_info"]
                for sid, segment in enumerate(segment_info):
                    if segment[2] == target_state:
                        readout_meth = segment[1].keys()[0]
                        v_ranges = segment[1][readout_meth]
                        axis_keys = ["range_x", "range_y"]
                        for plunger, key in zip(plungers, axis_keys):
                            plunger.current_valid_range(v_ranges[key])

                        tuningresult = self.get_charge_diagram(
                            device,
                            plungers,
                            voltage_precision=0.0001,
                        )
                        self.measurement_result_all[device.name].add_result(
                            tuningresult, f"chargediagram_{n_iter}_segment_{sid}"
                        )

            if continue_tuning:
                done = False
                logger.warning("Continue tuning regardless of the outcome.")

            if not done:
                self.update_gate_configuration(
                    device,
                    tuningresult,
                    target_state,
                )
                logger.info(
                    (
                        "Continuing with new configuration: "
                        f"{device.get_gate_status()}."
                    )
                )

        if n_iter >= max_iter:
            logger.info(f"Tuning {device.name}: Max number of iterations reached.")

        return self.measurement_result_all[device.name]

    def set_helper_gate(
        self,
        device: Device,
        helper_gate_id: int = 0,
        gates_for_init_range_measurement: Optional[Sequence[int]] = [1, 2, 4],
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
            raise ValueError("Unknown tuning outcome. Expect either a \
                successful tuning stage or termination reasons")

        if not termination_reasons and last_result.success:
            # A good regime was found, but not the right one.
            initial_update = self._update_dotregime_directive(
                target_state, central_barrier_id
            )
        elif termination_reasons and not last_result.success:
            # the charge diagram terminated unsuccessfully: no plunger ranges
            # were found to give a good diagram. The device might be too
            # pinched off or too open.
            initial_update = self._select_barrier_directives(
                termination_reasons, outer_barrier_ids,
            )
        else:
            raise NotImplementedError('Unknown tuning status')

        self.adjust_barriers_loop(
            device,
            target_state,
            initial_update,
            helper_gate_id,
            central_barrier_id,
            outer_barrier_ids,
            range_change_setting=range_change_setting,
            )

    def _update_dotregime_directive(
        self,
        target_regime: DeviceState,
        central_barrier_id: int = 3,
    ):
        print(target_regime)
        if target_regime == DeviceState.singledot:
            v_direction = VoltageChangeDirection.positive
            print('single')
        elif target_regime == DeviceState.doubledot:
            v_direction = VoltageChangeDirection.negative
            print('double')
        else:
            raise ValueError('Invalid target regime.')
        return {central_barrier_id: v_direction}

    def _select_barrier_directives(
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

    def adjust_barriers_loop(
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
        success = True if new_v_direction is None else False
        while not success:
            new_v_direction = self.adjust_all_barriers(
                device,
                target_state,
                new_v_direction,
                helper_gate_id,
                central_barrier_id,
                outer_barriers_id,
            )
            success = True if new_v_direction is None else False

    def adjust_all_barriers(
        self,
        device,
        target_state: DeviceState,
        voltage_change_direction: VoltageChangeDirection,
        helper_gate_id: int = 0,
        central_barrier_id: int = 3,
        outer_barriers_id: Sequence[int] = (1, 4),
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
    ):
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
        voltage_change_direction = self.set_outer_barriers(
            device,
            gate_ids=outer_barriers_id,
        )
        return voltage_change_direction

    def set_outer_barriers(
        self,
        device: Device,
        gate_ids: Optional[Sequence[float]] = (1, 4),
        main_readout_method: ReadoutMethods = ReadoutMethods.transport,
        tolerance: float = 0.1,
    ) -> Tuple[bool, VoltageChangeDirection]:
        """
        Will not update upper valid range limit. We assume it has been
        determined in the beginning with central_barrier = 0 V and does
        not change.
        new_voltage = T + 2 / 3 * abs(T - H)
        """
        if isinstance(gate_ids, int):
            gate_ids = [gate_ids]

        new_direction = None
        for barrier_id in gate_ids:
            barrier = device.gates[barrier_id]
            result = self.characterize_gate(
                device,
                barrier,
                comment="Characterize outer barriers before setting them.",
                use_safety_voltage_ranges=True,
            )
            features = result.ml_result["features"][main_readout_method.name]
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
        main_readout_method: ReadoutMethods = ReadoutMethods.transport,
        gate_id: int = 3,
    ) -> None:
        """"""
        assert isinstance(target_state, DeviceState)
        barrier = device.gates[gate_id]
        result = self.characterize_gate(
            device,
            barrier,
            use_safety_voltage_ranges=True,
        )
        features = result.ml_result["features"][main_readout_method.name]

        if target_state == DeviceState.singledot:
            barrier.voltage(features["high_voltage"])
        elif target_state == DeviceState.doubledot:
            data_id = result.data_ids[-1]
            ds = nt.Dataset(
                data_id,
                self.data_settings.db_name,
                db_folder=self.data_settings.db_folder,
            )
            signal = ds.data[main_readout_method.name].values
            voltage = ds.data[main_readout_method.name]["voltage_x"].values

            v_sat_idx = np.argwhere(signal < float(2 / 3))
            if len(v_sat_idx) > 0:
                v_index = v_sat_idx[-1][0]
            else:
                v_index = 0
            barrier.voltage(voltage[v_index])
        elif target_state == DeviceState.pinchedoff:
            barrier.voltage(features["low_voltage"])
        else:
            raise ValueError("Invalid DeviceState. Use singledot, doubledot \
                or pinchedoff.")

        new_range = [features["low_voltage"], features["high_voltage"]]
        device.current_valid_ranges({barrier.gate_id: new_range})
        logger.info(f"Central barrier set to {barrier.voltage()} with valid \
            range ({features['low_voltage']}, {features['high_voltage']}). ")

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
        noise_floor: float = 0.02,
        open_signal: float = 0.1,
        main_readout_method: ReadoutMethods = ReadoutMethods.transport,
        plunger_barrier_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[bool, Dict[int, str]]:
        """
        noise_floor and open_signal compared to normalised signal
        checks if barriers need to be adjusted, depending on min and max signal
        """
        if plunger_barrier_pairs is None:
            plunger_barrier_pairs = [(2, 1), (4, 5)]
        if not isinstance(plunger_barrier_pairs, list):
            raise ValueError('Invalid plunger_barrier_pairs input.')

        check_readout_method(device, main_readout_method)
        barrier_changes: Dict[int, VoltageChangeDirection] = {}
        success = True

        for plunger_id, barrier_id in plunger_barrier_pairs:
            plunger = device.gates[plunger_id]
            new_range, device_state = self.characterize_plunger(
                device, plunger, main_readout_method, noise_floor, open_signal
            )
            device.current_valid_ranges({plunger_id: new_range})

            if device_state == DeviceState.pinchedoff:
                drct = VoltageChangeDirection.positive
                barrier_changes[barrier_id] = drct
                success = False

            if device_state == DeviceState.opencurrent:
                drct = VoltageChangeDirection.negative
                barrier_changes[barrier_id] = drct
                success = False

        return success, barrier_changes

    def characterize_plunger(
        self,
        device: Device,
        plunger: DeviceChannel,
        main_readout_method: ReadoutMethods = ReadoutMethods.transport,
        noise_floor: float = 0.02,
        open_signal: float = 0.1,
    ) -> DeviceState:
        """ """
        check_readout_method(device, main_readout_method)
        result = self.characterize_gate(
            device,
            plunger,
            use_safety_voltage_ranges=True,
            iterate=False,
        )
        features = result.ml_result["features"][main_readout_method.name]
        new_range = (features["low_voltage"], features["high_voltage"])

        device_state = DeviceState.undefined
        if features["min_signal"] > open_signal:
            device_state = DeviceState.opencurrent
        if features["max_signal"] < noise_floor:
            device_state = DeviceState.pinchedoff

        return new_range, device_state

    def choose_new_gate_voltage(
        self,
        device: Device,
        gate_id: int,
        voltage_change_direction: VoltageChangeDirection,
        range_change_setting: RangeChangeSetting = RangeChangeSetting(),
        # relative_range_change: float = 0.5,
        # max_range_change: float = 0.1,
        # min_range_change: float = 0.05,
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


def check_readout_method(device, readout_method):
    if not isinstance(readout_method, ReadoutMethods):
        raise ValueError("Unknown main readout method.")
    if getattr(device.readout, readout_method.name) is None:
        raise ValueError(
            f'Main readout method {readout_method} not found for {device}'
        )


