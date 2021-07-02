import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import nanotune as nt
from nanotune.device.device import Device, ReadoutMethods, ReadoutMethodsLiteral
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
        desired_regime: str = "doubledot",
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
            desired_regime=desired_regime,
            max_iter=max_iter,
            set_barriers=True,
            take_high_res=take_high_res,
        )
        return result

    def tune_1D(
        self,
        device: Device,
        desired_regime: str,
        max_iter: int = 100,
        set_barriers: bool = True,
        take_high_res: bool = False,
        reset_gates: bool = False,
        continue_tuning: bool = False,
    ) -> Dict[Any, Any]:
        """Does not reset any tuning"""

        if reset_gates:
            device.all_gates_to_highest()

        done = False
        if set_barriers:
            self.set_central_barrier(device, desired_regime=desired_regime)
            (success,
             new_voltage_change_directions) = self.set_outer_barriers(device)
            while not success:
                self.update_top_barrier(device, new_voltage_change_directions)
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
                 barrier_change_directions) = self.set_new_plunger_ranges(
                    device
                )

                if not success:
                    out_br_success, new_v_change_dir = self.update_barriers(
                        device, barrier_change_directions
                    )

                    if not out_br_success:
                        logger.info(
                            (
                                "Outer barrier reached safety limit. "
                                "Setting new top barrier.\n"
                            )
                        )
                        self.update_top_barrier(device, new_v_change_dir)
                        (success,
                         new_v_change_dir) = self.set_outer_barriers(device)
                        while not success:
                            self.update_top_barrier(device, new_v_change_dir)
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
            if desired_regime in segment_info[:, 2]:
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
                    if segment[2] == desired_regime:
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
                    desired_regime,
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
        helper_gate_id: int = 0,  # top_barrier in 2D doubledot device
        gates_for_init_range_measurement: Optional[Sequence[int]] = [1, 2, 4],
    ) -> None:
        """ """
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
        desired_regime: str,
        # range_change: float = 0.1,
        # min_change: float = 0.01,
    ) -> None:
        """
        Choose new gate voltages when previous tuning did not result in any
        good regime.
        """
        termination_reasons = last_result.termination_reasons

        if not termination_reasons and not last_result.success:
            raise ValueError(
                (
                    "Unknown tuning outcome. Expect either a successful "
                    "tuning stage or termination reasons"
                )
            )

        if not termination_reasons:
            # A good regime was found, but not the right one. Change central
            # barrier to get there
            if desired_regime == 1:
                v_direction = VoltageChangeDirection.positive
            else:
                v_direction = VoltageChangeDirection.negative
            success, new_v_direction = self.update_barriers(
                device,
                {device.central_barrier.gate_id: v_direction},
                range_change=0.05,
            )
            while not success:
                self.update_top_barrier(device, new_v_direction)
                self.set_central_barrier(device, desired_regime=desired_regime)
                success, new_v_direction = self.set_outer_barriers(device)
        else:
            # the charge diagram terminated unsuccessfully: no plunger ranges
            # were found to give a good diagram. The device might be too
            # pinched off or too open.
            possible_changes = [
                (device.left_barrier, VoltageChangeDirection.positive),
                (device.left_barrier, VoltageChangeDirection.negative),
                (device.right_barrier, VoltageChangeDirection.positive),
                (device.right_barrier, VoltageChangeDirection.negative),
            ]
            barrier_change_dir = {}
            for gate, change_dir in possible_changes:
                if change_dir in termination_reasons:
                    barrier_change_dir[gate.gate_id] = change_dir

            success, new_v_change_dir = self.update_barriers(
                device,
                barrier_change_dir,
            )
            while not success:
                self.update_top_barrier(device, new_v_change_dir)
                self.set_central_barrier(device, desired_regime=desired_regime)
                success, new_v_change_dir = self.set_outer_barriers(device)

    def update_top_barrier(
        self,
        device: Device,
        voltage_change_direction: VoltageChangeDirection,
    ) -> None:
        """ """
        new_top = self.choose_new_gate_voltage(
            device,
            0,
            voltage_change_direction,
            range_change=0.1,
            max_range_change=0.05,
            min_range_change=0.01,
        )
        device.top_barrier.voltage(new_top)
        logger.info(
            (
                f"Setting top barrier to {new_top}"
                "Characterize and set outer barriers again.\n"
            )
        )

    def set_outer_barriers(
        self,
        device: Device,
        main_readout_method: ReadoutMethodsLiteral = 'transport',
    ) -> Tuple[bool, str]:
        """
        Will not update upper valid range limit. We assume it has been
        determined in the beginning with central_barrier = 0 V and does
        not change.
        new_voltage = T + 2 / 3 * abs(T - H)
        """
        # (barrier_values, success) = self.get_outer_barriers()
        # result = self.characterize_gate(
        #     device,
        #     gates=[device.left_barrier, device.right_barrier],
        #     comment="Characterize outer barriers before setting them.",
        #     use_safety_voltage_ranges=True,
        # )
        success = True

        for barrier in [device.left_barrier, device.right_barrier]:
            result = self.characterize_gate(
                device,
                barrier,
                comment="Characterize outer barriers before setting them.",
                use_safety_voltage_ranges=True,
            )
            # key = f"characterization_{barrier.name}"
            # features = result.tuningresults[key].features
            features = result.ml_result["features"][main_readout_method]

            L, H = features["low_voltage"], features["high_voltage"]
            T = features["transition_voltage"]

            new_voltage = T + 2 / 3 * abs(T - H)

            curr_sft = barrier.safety_voltage_range()

            touching_limits = np.isclose(new_voltage, curr_sft, atol=0.005)
            if any(touching_limits):
                success = False
                if touching_limits[0]:
                    new_direction = VoltageChangeDirection.negative
                else:
                    new_direction = VoltageChangeDirection.positive
            else:
                barrier.voltage(new_voltage)
                barrier.current_valid_range((L, H))
                barrier.transition_voltage(T)
                logger.info(f"Setting {barrier.name} to {new_voltage}.")

        return success, new_direction

    def set_central_barrier(
        self,
        device: Device,
        desired_regime: str = "doubledot",
        main_readout_method: ReadoutMethodsLiteral = 'transport',
    ) -> None:
        """"""
        setpoint_settings = copy.deepcopy(self.data_settings)
        setpoint_settings.parameters_to_sweep = [device.central_barrier]

        result = self.characterize_gate(
            device,
            device.central_barrier,
            use_safety_voltage_ranges=True,
        )
        features = result.ml_result["features"][main_readout_method]

        if desired_regime == "singledot":
            device.central_barrier.voltage(features["high_voltage"])
        elif desired_regime == "doubledot":
            data_id = result.data_ids[-1]
            ds = nt.Dataset(
                data_id,
                self.data_settings.db_name,
                db_folder=self.data_settings.db_folder,
            )

            signal = ds.data[main_readout_method].values
            voltage = ds.data[main_readout_method]["voltage_x"].values

            v_sat_idx = np.argwhere(signal < float(2 / 3))[-1][0]
            three_quarter = voltage[v_sat_idx]

            device.central_barrier.voltage(three_quarter)
        else:
            raise ValueError("Unknown regime. Try 'singledot' or 'doubledot'.")

        device.current_valid_ranges({
            device.central_barrier.gate_id: [
                features["low_voltage"], features["high_voltage"]
                ]
            }
        )

        logger.info(f"Central barrier to {device.central_barrier.voltage()}")

    def update_barriers(
        self,
        device: Device,
        barrier_change_directions: Dict[int, VoltageChangeDirection],
        relative_range_change: float = 0.1,
    ) -> Optional[VoltageChangeDirection]:
        """
        returns new direction if touching limits
        In that case some other gate needs to be set more negative.
        """
        for gate_id, direction in barrier_change_directions.items():
            barrier = device.gates[gate_id]
            new_voltage = self.choose_new_gate_voltage(
                device,
                gate_id,
                direction,
                relative_range_change=relative_range_change,
                max_range_change=0.05,
                min_range_change=0.01,
            )
            safe_range = barrier.safety_voltage_range()
            touching_limits = np.isclose(new_voltage, safe_range, atol=0.1)
            new_direction = None
            if touching_limits[0]:
                new_direction = VoltageChangeDirection.negative
            if touching_limits[1]:
                new_direction = VoltageChangeDirection.positive
            else:
                device.gates[gate_id].voltage(new_voltage)
                logger.info(
                    f"Choosing new voltage for {barrier}: {new_voltage}"
                )

        return new_direction

    def set_new_plunger_ranges(
        self,
        device: Device,
        noise_floor: float = 0.02,
        open_signal: float = 0.1,
        main_readout_method: ReadoutMethodsLiteral = 'transport',
        plunger_barrier_pairs: Sequence[Tuple[int, int]] = [(2, 1), (4, 5)],
    ) -> Tuple[bool, Dict[int, str]]:
        """
        noise_floor and open_signal compared to normalised signal
        checks if barriers need to be adjusted, depending on min and max signal
        """
        check_readout_method(device, main_readout_method)
        barrier_change_directions: Dict[int, VoltageChangeDirection] = {}
        success = True

        for plunger_id, barrier_id in plunger_barrier_pairs:
            plunger = device.gates[plunger_id]
            new_range, device_state = self.characterize_plunger(
                device, plunger, main_readout_method, noise_floor, open_signal
            )
            device.current_valid_ranges({plunger_id: new_range})

            if device_state == DeviceState.pinchedoff:
                drct = VoltageChangeDirection.positive
                barrier_change_directions[barrier_id] = drct
                success = False

            if device_state == DeviceState.opencurrent:
                drct = VoltageChangeDirection.negative
                barrier_change_directions[barrier_id] = drct
                success = False

        return success, barrier_change_directions

    def characterize_plunger(
        self,
        device: Device,
        plunger: DeviceChannel,
        main_readout_method: ReadoutMethodsLiteral = 'transport',
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
        features = result.ml_result["features"][main_readout_method]
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
        relative_range_change: float = 0.5,
        max_range_change: float = 0.1,
        min_range_change: float = 0.05,
    ) -> float:
        """
        based on current_valid_range or safety_voltage_range if no current_valid_range
        is set
        """
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


def check_readout_method(device, readout_method):
    if not readout_method in ReadoutMethods.__dataclass_fields__.keys():
        raise ValueError("Unknown main readout method.")
    if getattr(device.readout, readout_method) is None:
        raise ValueError(
            f'Main readout method {readout_method} not found for {device}'
        )


