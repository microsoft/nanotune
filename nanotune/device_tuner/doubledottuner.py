import os
import copy
import logging
import time
import datetime
from typing import List, Optional, Dict, Tuple, Sequence, Callable, Any, Union
from functools import partial
import numpy as np

import qcodes as qc
from qcodes import validators as vals
from qcodes.dataset.experiment_container import (load_last_experiment,
                                                 load_experiment)

import nanotune as nt
from nanotune.device.device import Device as Nt_Device
from nanotune.fit.pinchofffit import PinchoffFit
from nanotune.tuningstages.chargediagram import ChargeDiagram
from nanotune.classification.classifier import Classifier
from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.device.gate import Gate
from nanotune.utils import flatten_list
from nanotune.device_tuner.tuner import Tuner

logger = logging.getLogger(__name__)
DATA_DIMS = {
    'gatecharacterization1d': 1,
    'chargediagram': 2,
    'coulomboscillations': 1,
}


class DoubleDotTuner(Tuner):
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
    fit_options = {
        'pinchofffit': Dict[str, Any],
        'dotfit': Dict[str, Any],
    }
    """
    def __init__(
        self,
        name: str,
        data_settings: Dict[str, Any],
        classifiers: Dict[str, Classifier],
        setpoint_settings: Dict[str, Any],
        fit_options: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            name,
            data_settings,
            classifiers,
            setpoint_settings,
            fit_options=fit_options,
            )
        self.tuningresults = {}

    def tune(
        self,
        device: Nt_Device,
        desired_regime: str = 'doubledot',
        max_iter: int = 100,
        take_high_res: bool = False,
        ) -> TuningResult:
        """
        """
        assert 'singledot' in self.classifiers.keys()
        assert 'doubledot' in self.classifiers.keys()
        assert 'dotregime' in self.classifiers.keys()

        if device.name not in self.tuningresults.keys():
            self.tuningresults[device.name] = []

        if self.qcodes_experiment.sample_name != Nt_Device.name:
            logger.warning(
                "The device's name does match the"
                + " the sample name in qcodes experiment."
            )

        device.all_gates_to_highest()
        self.update_normalization_constants(device)
        self.set_top_barrier(device)

        tuningresult = self.tune_1D(
            device,
            desired_regime=desired_regime,
            max_iter=max_iter,
            set_barriers=True,
            take_high_res=take_high_res,
        )
        return tuningresult

    def set_top_barrier(self,
        device: Nt_Device,
        ) -> None:
        """ """
        if device.initial_valid_ranges()[0] == device.gates[0].safety_range():
            (top_barrier_ranges,
            tuningresult) = self.measure_initial_ranges(
                device.gates[0], [device.gates[1], device.gates[5]]
            )
            self.tuningresults[device.name].append(tuningresult)
        else:
            top_barrier_ranges = device.initial_valid_ranges()[0]
            # L, H = top_barrier_ranges
            # tb_voltage = 0.5 * (H + T)

        device.gates[0].current_valid_range(top_barrier_ranges)
        device.gates[0].dc_voltage(top_barrier_ranges[1])

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
        if device.name not in self.tuningresults.keys():
            self.tuningresults[device.name] = []
        self.tuningresults[device.name].append(TuningResult(device.name))

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
                            ('Outer barrier reached safety limit. '
                            'Setting new top barrier.\n')
                        )
                        self.update_top_barrier(device, new_action)
                        success, new_action = self.set_outer_barriers(device)
                        while not success:
                            self.update_top_barrier(device, new_action)
                            (success,
                             new_action) = self.set_outer_barriers(device)
                else:
                    good_barriers = True

            tuningresult = self.get_charge_diagram(
                device,
                plungers,
                signal_thresholds=[0.004, 0.1],
            )
            self.tuningresults[device.name].append(tuningresult)
            stage_summary = tuningresult.stage_summaries['chargediagram']
            logger.info(
                (f'ChargeDiagram stage finished: {stage_summary["success"]}\n'
                f'termination_reason: {stage_summary["termination_reasons"]}\n')
            )
            segment_info = stage_summary['stage_summary']
            if desired_regime in segment_info[:, 2]:
                logger.info('Desired regime found.')
                done = True

            if done and take_high_res:
                segment_info = stage_summary['segment_info']

                logger.info("Take high resolution of entire charge diagram.")
                tuningresult = self.get_charge_diagram(
                    device,
                    plungers,
                    voltage_precision=0.0005,
                    update_settings=False,
                )
                self.tuningresults[device.name].append(tuningresult)
                logger.info("Take high resolution of charge diagram segments.")
                for segment in segment_info:
                    if segment[2] == desired_regime:
                        readout_meth = segment[1].keys()[0]
                        v_ranges = segment[1][readout_meth]
                        axis_keys = ['range_x', 'range_y']
                        for plunger, key in zip(plungers, axis_keys):
                            plunger.current_valid_range(v_ranges[key])

                        tuningresult = self.get_charge_diagram(
                            device,
                            plungers,
                            voltage_precision=0.0001,
                            update_settings=False,
                        )
                        self.tuningresults[device.name].append(tuningresult)

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
                    ('Continuing with new configuration: '
                    f'{device.get_gate_status()}.')
                    )

        if n_iter >= max_iter:
            logger.info(
                f'Tuning {device.name}: Max number of iterations reached.'
                )

        return self.tuningresults

    def update_top_barrier(self, device: Nt_Device, new_action: str) -> None:
        """ """
        new_top = self.choose_new_gate_voltage(
                            device,
                            0,
                            new_action,
                            range_change=0.1,
                            max_range_change=0.05,
                            min_range_change=0.01,
                        )
        device.top_barrier.dc_voltage(new_top)
        logger.info(
            (f'Setting top barrier to {new_top}'
            'Characterize and set outer barriers again.\n')
            )

    def update_barriers(
        self,
        device: Nt_Device,
        barrier_actions: Dict[int, List[str]],
        range_change: float = 0.1,
        ) -> Tuple[bool, Dict[int, List[float]]]:
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
                        new_action = 'more negative'
                    else:
                        new_action = 'more positive'
                else:
                    device.gates[gate_layout_id].dc_voltage(new_voltage)
                    logger.info(
                        ('Choosing new voltage range for'
                         f'{device.gates[gate_layout_id.name]}: {new_voltage}')
                    )

        return success, new_action

    def set_outer_barriers(
        self,
        device: Nt_Device,
    ) -> Tuple[bool, Dict[int, List[float]]]:
        """
        Will not update upper valid range limit. We assume it has been
        determined in the beginning with central_barrier = 0 V and does
        not change.
        new_voltage = T + 2 / 3 * abs(T - H)
        """
        # (barrier_values, success) = self.get_outer_barriers()
        sub_result = self.characterize_gates(
            device,
            gates=[device.left_barrier, device.right_barrier],
            comment='Characetize outer barriers before setting them.',
            use_safety_ranges=True,
        )
        self.tuningresults[device.name].append(sub_result)
        stage_summaries = sub_result.stage_summaries
        success = True

        for barrier in [device.left_barrier, device.right_barrier]:
            key = f"characterization_{barrier.name}"
            features = stage_summaries[key]['features']

            L, H = features['low_voltage'], features['high_voltage']
            T = features['transition_voltage']

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
                barrier.dc_voltage(new_voltage)
                barrier.current_valid_range((L, H))
                barrier.transition_voltage(T)
                logger.info(f'Setting {barrier.name} to {new_voltage}.')

        return success, new_action


    def set_central_barrier(
        self,
        device: Nt_Device,
        desired_regime: str = 'doubledot',
    ) -> None:
        """"""
        setpoint_settings = copy.deepcopy(self.setpoint_settings())
        setpoint_settings['gates_to_sweep'] = [device.central_barrier]

        sub_result = self.characterize_gates(
                        device,
                        gates=[device.central_barrier],
                        use_safety_ranges=True,
                        )
        self.tuningresults[device.name].append(sub_result)
        key = f"characterization_{device.central_barrier.name}"
        stage_summary = sub_result.stage_summaries[key]
        features = stage_summary['features']

        if desired_regime == 1:
            self.central_barrier.dc_voltage(features['high_voltage'])
        elif desired_regime == 3:
            data_id = stage_summary['data_ids'][-1]
            ds = nt.Dataset(
                data_id, self.data_settings['db_name'],
                db_folder=self.data_settings['db_folder'],
                )
            read_meths = device.readout_methods().keys()
            if 'dc_current' in read_meths:
                signal = ds.data['dc_current'].values
                voltage = ds.data['dc_current']['voltage_x'].values
            elif 'dc_sensor' in read_meths:
                signal = ds.data['dc_sensor'].values
                voltage = ds.data['dc_sensor']['voltage_x'].values
            elif 'rf' in read_meths:
                signal = ds.data['rf'].values
                voltage = ds.data['rf']['voltage_x'].values
            else:
                raise ValueError('Unknown readout method.')
            v_sat_idx = np.argwhere(signal < float(2 / 3))[-1][0]
            three_quarter = voltage[v_sat_idx]

            self.central_barrier.dc_voltage(three_quarter)
        else:
            raise ValueError

        self.central_barrier.current_valid_range(
            [features['low_voltage'], features['high_voltage']]
            )

        logger.info(
            f'Central barrier to {self.central_barrier.dc_voltage()}'
        )

    def update_gate_configuration(
        self,
        device: Nt_Device,
        tuningresult: TuningResult,
        desired_regime: str,
        # range_change: float = 0.1,
        # min_change: float = 0.01,
    ) -> None:
        """
        Choose new gate voltages when previous tuning did not result in any good
        regime.
        """
        stage_summary = tuningresult.stage_summaries['chargediagram']
        termination_reasons = stage_summary['termination_reasons']

        if not termination_reasons and not stage_summary['success']:
            raise ValueError(
                ('Unknown tuning outcome. Expect either a successful '
                'tuning stage or termination reasons')
                )

        if not termination_reasons:
            # A good regime was found, but not the right one. Change central
            # barrier to get there
            if desired_regime == 1:
                action = 'more positive'
            else:
                action = 'more negative'
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
            # were found to give a good diagram. The device might be too pinched
            # off or too open.
            all_actions = [
            "x more negative", "x more positive",
            "y more negative", "y more positive"
            ]
            gates_to_change = [
                device.left_barrier, device.left_barrier,
                device.right_barrier, device.right_barrier]
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
        curr_v = device.gates[layout_id].dc_voltage()
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
            raise ValueError('Invalid action in choose_new_gate_voltage')

        return new_v

    def set_new_plunger_ranges(
        self,
        device: Nt_Device,
        noise_floor: float = 0.02,
        open_signal = 0.1,
    ) -> None:
        """
        noise_floor and open_signal compared to normalised signal
        checks if barriers need to be adjusted, depending on min and max signal
        """
        sub_result = self.characterize_gates(
                        device,
                        gates=[device.left_plunger, device.right_plunger],
                        use_safety_ranges=True,
                        )
        plunger_barrier_pairs = {
            device.left_plunger: device.left_barrier,
            device.right_plunger: device.right_barrier,
        }
        self.tuningresults[device.name].append(sub_result)

        barrier_actions: Dict[int, List[str]] = {}
        success = True
        for plunger, barrier in plunger_barrier_pairs.items():
            key = f"characterization_{plunger.name}"
            features = sub_result.stage_summaries[key]['features']

            max_sig = features['max_signal']
            min_sig = features['min_signal']
            low_voltage = features['low_voltage']
            high_voltage = features['low_voltage']
            new_range = (low_voltage, high_voltage)
            plunger.current_valid_range(new_range)
            logger.info(
                f"{plunger.name}: new current valid range set to {new_range}"
            )
            barrier_actions[barrier.layout_id()] = []
            if min_sig > open_signal:
                barrier_actions[barrier.layout_id()].append("more negative")
                success = False
            if max_sig < noise_floor:
                barrier_actions[barrier.layout_id()].append("more positive")
                success = False

        return success, barrier_actions

    def get_charge_diagram(
        self,
        device: Nt_Device,
        gates_to_sweep: List[Gate],
        voltage_precision: Optional[float] = None,
        signal_thresholds: Optional[List[float]] = None,
        update_settings: bool = True,
        comment: str = '',
    ) -> Tuple[
        bool,
        List[int],
        Dict[Any, Any],
        List[Tuple[int, Tuple[Tuple[float, float], Tuple[float, float]], int]],
    ]:
        """
        stage.segment_info = ((data_id, ranges, category))
        """
        required_clf = ['singledot', 'doubledot', 'dotregime']
        for clf in required_clf:
            if clf not in self.classifiers.keys():
                raise KeyError(f'No {clf} classifier found.')

        setpoint_settings = copy.deepcopy(self.setpoint_settings())
        setpoint_settings['gates_to_sweep'] = gates_to_sweep
        setpoint_settings['voltage_precision'] = voltage_precision

        tuningresult = TuningResult(device.name)

        with self.device_specific_settings(device):
            self.fit_options({'dotfit':
                {'signal_thresholds': signal_thresholds}
                })
            stage = ChargeDiagram(
                data_settings=self.data_settings,
                setpoint_settings=setpoint_settings,
                readout_methods=device.readout_methods(),
                fit_options=self.fit_options()['dotfit'],
                update_settings=update_settings,
                classifiers=self.classifiers,
                measurement_options=device.measurement_options(),
            )

            success, termination_reasons, result = stage.run_stage()

            result['device_gates_status'] = device.get_gate_status()
            result['segment_info'] = stage.segment_info
            names = [gate.name for gate in gates_to_sweep]
            tuningresult.add_result(
                f"charge_diagram_{names.join('_')}", success,
                termination_reasons, result,
                comment,
                )

        return tuningresult


