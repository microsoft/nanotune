import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import qcodes as qc
from qcodes import validators as vals
from qcodes.dataset.experiment_container import (load_experiment,
                                                 load_last_experiment)

import nanotune as nt
from nanotune.classification.classifier import Classifier
from nanotune.device.device import Device as Nt_Device
from nanotune.device.device_channel import DeviceChannel
from nanotune.device_tuner.tuner import Tuner
from nanotune.device_tuner.tuningresult import MeasurementHistory, TuningResult
from nanotune.fit.pinchofffit import PinchoffFit
from nanotune.tuningstages.chargediagram import ChargeDiagram
from nanotune.utils import flatten_list

logger = logging.getLogger(__name__)
DATA_DIMS = {
    "gatecharacterization1d": 1,
    "chargediagram": 2,
    "coulomboscillations": 1,
}


class DotTuner(Tuner):
    """
    classifiers = {
        'pinchoff': Optional[Classifier],
        'singledot': Optional[Classifier],
        'doubledot': Optional[Classifier],
        'dotregime': Optional[Classifier],
    }
    data_settings = {
        'db_name': str,
        'db_folder': Optional[str],
        'qc_experiment_id': Optional[int],
        'segment_db_name': Optional[str],
        'segment_db_folder': Optional[str],
    }
    setpoint_settings = {
        'voltage_precision': float,
    }
    """

    def __init__(
        self,
        name: str,
        data_settings: Dict[str, Any],
        classifiers: Dict[str, Classifier],
        setpoint_settings: Dict[str, Any],
    ) -> None:
        super().__init__(
            name,
            data_settings,
            classifiers,
            setpoint_settings,
        )
        self._tuningresults_all: Dict[str, MeasurementHistory] = {}

    def tune(
        self,
        device: Nt_Device,
        desired_regime: str = "doubledot",
        max_iter: int = 100,
        take_high_res: bool = False,
    ) -> MeasurementHistory:
        """"""
        assert "singledot" in self.classifiers.keys()
        assert "doubledot" in self.classifiers.keys()
        assert "dotregime" in self.classifiers.keys()

        if device.name not in self._tuningresults_all.keys():
            self._tuningresults_all[device.name] = MeasurementHistory(device.name)

        if self.qcodes_experiment.sample_name != Nt_Device.name:
            logger.warning(
                "The device's name does match the"
                + " the sample name in qcodes experiment."
            )

        device.all_gates_to_highest()
        self.update_normalization_constants(device)
        self.set_top_barrier(device)

        result = self.tune_1D(
            device,
            desired_regime=desired_regime,
            max_iter=max_iter,
            set_barriers=True,
            take_high_res=take_high_res,
        )
        return result

    def set_top_barrier(
        self,
        device: Nt_Device,
    ) -> None:
        """ """
        if device.initial_valid_ranges()[0] == device.gates[0].safety_range():
            (top_barrier_ranges, measurement_result) = self.measure_initial_ranges(
                device.gates[0], [device.gates[1], device.gates[5]]
            )
            self._tuningresults_all[device.name].update(measurement_result)
        else:
            top_barrier_ranges = device.initial_valid_ranges()[0]
            # L, H = top_barrier_ranges
            # tb_voltage = 0.5 * (H + T)

        device.gates[0].current_valid_range(top_barrier_ranges)
        device.gates[0].voltage(top_barrier_ranges[1])

    def tune_1D(
        self,
        device: Nt_Device,
        desired_regime: str,
        max_iter: int = 100,
        set_barriers: bool = True,
        take_high_res: bool = False,
        reset_gates: bool = False,
        continue_tuning: bool = False,
    ) -> Dict[Any, Any]:
        """Does not reset any tuning"""
        if device.name not in self._tuningresults_all.keys():
            self._tuningresults_all[device.name] = MeasurementHistory(device.name)

        if reset_gates:
            device.all_gates_to_highest()

        done = False
        if set_barriers:
            self.set_central_barrier(device, desired_regime=desired_regime)
            success, new_action = self.set_outer_barriers(device)
            while not success:
                self.update_top_barrier(device, new_action)
                success, new_action = self.set_outer_barriers(device)

        plungers = [device.left_plunger, device.right_plunger]
        n_iter = 0
        while not done and n_iter <= max_iter:
            n_iter += 1
            good_barriers = False

            while not good_barriers:
                # Narrow down plunger ranges:
                success, barrier_actions = self.set_new_plunger_ranges(device)

                if not success:
                    out_br_success, new_action = self.update_barriers(
                        device, barrier_actions
                    )

                    if not out_br_success:
                        logger.info(
                            (
                                "Outer barrier reached safety limit. "
                                "Setting new top barrier.\n"
                            )
                        )
                        self.update_top_barrier(device, new_action)
                        success, new_action = self.set_outer_barriers(device)
                        while not success:
                            self.update_top_barrier(device, new_action)
                            (success, new_action) = self.set_outer_barriers(device)
                else:
                    good_barriers = True

            tuningresult = self.get_charge_diagram(
                device,
                plungers,
                signal_thresholds=[0.004, 0.1],
            )
            self._tuningresults_all[device.name].add_result(
                tuningresult, f"chargediagram_{n_iter}"
            )
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
                self._tuningresults_all[device.name].add_result(
                    tuningresult, f"chargediagram_highres_{n_iter}"
                )
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
                        self._tuningresults_all[device.name].add_result(
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

        return self._tuningresults_all[device.name]

    def update_top_barrier(
        self,
        device: Nt_Device,
        new_action: str,
    ) -> None:
        """ """
        new_top = self.choose_new_gate_voltage(
            device,
            0,
            new_action,
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

    def update_barriers(
        self,
        device: Nt_Device,
        barrier_actions: Dict[int, str],
        range_change: float = 0.1,
    ) -> Tuple[bool, str]:
        """ """
        success = True
        for gate_layout_id, actions in barrier_actions.items():
            for action in actions:
                new_voltage = self.choose_new_gate_voltage(
                    device,
                    gate_layout_id,
                    action,
                    range_change=range_change,
                    max_range_change=0.05,
                    min_range_change=0.01,
                )
                safe_range = device.gates[gate_layout_id].safety_range()
                touching_limits = np.isclose(new_voltage, safe_range, atol=0.1)
                if any(touching_limits):
                    success = False
                    if touching_limits[0]:
                        # Some other gate needs to be set more negative
                        new_action = "more negative"
                    else:
                        new_action = "more positive"
                else:
                    device.gates[gate_layout_id].voltage(new_voltage)
                    logger.info(
                        (
                            "Choosing new voltage range for"
                            f"{device.gates[gate_layout_id]}: {new_voltage}"
                        )
                    )

        return success, new_action

    def set_outer_barriers(
        self,
        device: Nt_Device,
    ) -> Tuple[bool, str]:
        """
        Will not update upper valid range limit. We assume it has been
        determined in the beginning with central_barrier = 0 V and does
        not change.
        new_voltage = T + 2 / 3 * abs(T - H)
        """
        # (barrier_values, success) = self.get_outer_barriers()
        result = self.characterize_gates(
            device,
            gates=[device.left_barrier, device.right_barrier],
            comment="Characetize outer barriers before setting them.",
            use_safety_ranges=True,
        )
        self._tuningresults_all[device.name].update(result)
        success = True

        for barrier in [device.left_barrier, device.right_barrier]:
            key = f"characterization_{barrier.name}"
            features = result.tuningresults[key].features

            L, H = features["low_voltage"], features["high_voltage"]
            T = features["transition_voltage"]

            new_voltage = T + 2 / 3 * abs(T - H)

            curr_sft = barrier.safety_range()

            touching_limits = np.isclose(new_voltage, curr_sft, atol=0.005)
            if any(touching_limits):
                success = False
                if touching_limits[0]:
                    new_action = "more negative"
                else:
                    new_action = "more positive"
            else:
                barrier.voltage(new_voltage)
                barrier.current_valid_range((L, H))
                barrier.transition_voltage(T)
                logger.info(f"Setting {barrier.name} to {new_voltage}.")

        return success, new_action

    def set_central_barrier(
        self,
        device: Nt_Device,
        desired_regime: str = "doubledot",
    ) -> None:
        """"""
        setpoint_settings = copy.deepcopy(self.setpoint_settings())
        setpoint_settings["gates_to_sweep"] = [device.central_barrier]

        result = self.characterize_gates(
            device,
            gates=[device.central_barrier],
            use_safety_ranges=True,
        )
        self._tuningresults_all[device.name].update(result)
        key = f"characterization_{device.central_barrier.name}"
        features = result.tuningresults[key].features

        if desired_regime == 1:
            self.central_barrier.voltage(features["high_voltage"])
        elif desired_regime == 3:
            data_id = result.tuningresults[key].data_ids[-1]
            ds = nt.Dataset(
                data_id,
                self.data_settings["db_name"],
                db_folder=self.data_settings["db_folder"],
            )
            read_meths = device.readout_methods().keys()
            if "transport" in read_meths:
                signal = ds.data["transport"].values
                voltage = ds.data["transport"]["voltage_x"].values
            elif "sensing" in read_meths:
                signal = ds.data["sensing"].values
                voltage = ds.data["sensing"]["voltage_x"].values
            elif "rf" in read_meths:
                signal = ds.data["rf"].values
                voltage = ds.data["rf"]["voltage_x"].values
            else:
                raise ValueError("Unknown readout method.")
            v_sat_idx = np.argwhere(signal < float(2 / 3))[-1][0]
            three_quarter = voltage[v_sat_idx]

            self.central_barrier.voltage(three_quarter)
        else:
            raise ValueError

        self.central_barrier.current_valid_range(
            [features["low_voltage"], features["high_voltage"]]
        )

        logger.info(f"Central barrier to {self.central_barrier.voltage()}")

    def update_gate_configuration(
        self,
        device: Nt_Device,
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
                action = "more positive"
            else:
                action = "more negative"
            success, new_action = self.update_barriers(
                device,
                {device.central_barrier.layout_id(): action},
                range_change=0.05,
            )
            while not success:
                self.update_top_barrier(device, new_action)
                self.set_central_barrier(device, desired_regime=desired_regime)
                success, new_action = self.set_outer_barriers(device)
        else:
            # the charge diagram terminated unsuccessfully: no plunger ranges
            # were found to give a good diagram. The device might be too
            # pinched off or too open.
            all_actions = [
                "x more negative",
                "x more positive",
                "y more negative",
                "y more positive",
            ]
            gates_to_change = [
                device.left_barrier,
                device.left_barrier,
                device.right_barrier,
                device.right_barrier,
            ]
            barrier_actions = {}
            for action, gate in zip(all_actions, gates_to_change):
                if action in termination_reasons:
                    barrier_actions[gate.layout_id()] = action[2:]

            success, new_action = self.update_barriers(
                device,
                barrier_actions,
            )
            while not success:
                self.update_top_barrier(device, new_action)
                self.set_central_barrier(device, desired_regime=desired_regime)
                success, new_action = self.set_outer_barriers(device)

    def choose_new_gate_voltage(
        self,
        device: Nt_Device,
        layout_id: int,
        action: str,
        range_change: float = 0.5,
        max_range_change: float = 0.1,
        min_range_change: float = 0.05,
    ) -> float:
        """
        based on current_valid_range or safety_range if no current_valid_range
        is set
        """
        curr_v = device.gates[layout_id].voltage()
        sfty_rng = device.gates[layout_id].safety_range()
        try:
            L, H = device.gates[layout_id].current_valid_range()
        except ValueError:
            L, H = device.gates[layout_id].safety_range()

        if action == "more negative":
            v_change = max(range_change * abs(curr_v - L), min_range_change)
            v_change = min(max_range_change, v_change)
            new_v = curr_v - v_change
            if new_v < sfty_rng[0]:
                new_v = sfty_rng[0]
        elif action == "more positive":
            v_change = max(range_change * abs(curr_v - H), min_range_change)
            v_change = min(max_range_change, v_change)
            new_v = curr_v + v_change
            if new_v > sfty_rng[1]:
                new_v = sfty_rng[1]
        else:
            raise ValueError("Invalid action in choose_new_gate_voltage")

        return new_v

    def set_new_plunger_ranges(
        self,
        device: Nt_Device,
        noise_floor: float = 0.02,
        open_signal: float = 0.1,
    ) -> Tuple[bool, Dict[int, str]]:
        """
        noise_floor and open_signal compared to normalised signal
        checks if barriers need to be adjusted, depending on min and max signal
        """
        result = self.characterize_gates(
            device,
            gates=[device.left_plunger, device.right_plunger],
            use_safety_ranges=True,
        )
        plunger_barrier_pairs = {
            device.left_plunger: device.left_barrier,
            device.right_plunger: device.right_barrier,
        }
        self._tuningresults_all[device.name].update(result)

        barrier_actions: Dict[int, str] = {}
        success = True
        for plunger, barrier in plunger_barrier_pairs.items():
            key = f"characterization_{plunger.name}"
            features = result[key]["features"]

            max_sig = features["max_signal"]
            min_sig = features["min_signal"]
            low_voltage = features["low_voltage"]
            high_voltage = features["low_voltage"]
            new_range = (low_voltage, high_voltage)
            plunger.current_valid_range(new_range)
            logger.info(f"{plunger.name}: new current valid range set to {new_range}")
            if min_sig > open_signal:
                barrier_actions[barrier.layout_id()] = "more negative"
                success = False
            if max_sig < noise_floor:
                barrier_actions[barrier.layout_id()] = "more positive"
                success = False

        return success, barrier_actions

    def get_charge_diagram(
        self,
        device: Nt_Device,
        gates_to_sweep: List[DeviceChannel],
        voltage_precision: Optional[float] = None,
        signal_thresholds: Optional[List[float]] = None,
        iterate: bool = False,
        comment: str = "",
    ) -> TuningResult:
        """
        stage.segment_info = ((data_id, ranges, category))
        """
        required_clf = ["singledot", "doubledot", "dotregime"]
        for clf in required_clf:
            if clf not in self.classifiers.keys():
                raise KeyError(f"No {clf} classifier found.")

        setpoint_settings = copy.deepcopy(self.setpoint_settings())
        setpoint_settings["parameters_to_sweep"] = gates_to_sweep
        setpoint_settings["voltage_precision"] = voltage_precision

        with self.device_specific_settings(device):
            stage = ChargeDiagram(
                data_settings=self.data_settings,
                setpoint_settings=setpoint_settings,
                readout_methods=device.readout_methods(),
                classifiers=self.classifiers,
            )

            tuningresult = stage.run_stage(iterate=iterate)
            tuningresult.status = device.get_gate_status()
            tuningresult.comment = comment

        return tuningresult
