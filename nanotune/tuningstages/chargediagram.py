import logging
import json
from typing import Optional, Tuple, List, Union, Dict, Callable, Any

import numpy as np
import qcodes as qc
from qcodes.dataset.experiment_container import load_by_id

import nanotune as nt
from nanotune.tuningstages.tuningstage import TuningStage
from nanotune.device.gate import Gate
from nanotune.fit.dotfit import DotFit
from nanotune.classification.classifier import Classifier

logger = logging.getLogger(__name__)
DOT_LABLE_MAPPING = dict(nt.config["core"]["dot_mapping"])



class ChargeDiagram(TuningStage):
    """has a do_at_each for custom actions between setpoints"""

    def __init__(
        self,
        data_settings: Dict[str, Any],
        setpoint_settings: Dict[str, Any],
        readout_methods: Dict[str, qc.Parameter],
        classifiers: Dict[str, Classifier],
        measurement_options: Optional[Dict[str, Dict[str, Any]]] = None,
        fit_options: Optional[Dict[str, Any]] = None,
        update_settings: bool = True,
        range_change: float = 30,  # in percent
    ) -> None:
        """"""
        if 'segment_db_folder' not in data_settings.keys():
            data_settings['segment_db_folder'] = nt.config["db_folder"]
        if 'segment_db_name' not in data_settings.keys():
            seg_db_name = f'segmented_{nt.config["main_db"]}'
            data_settings['segment_db_name'] = seg_db_name
        if 'normalization_constants' not in data_settings.keys():
            logger.warning('No normalisation constants specified.')

        TuningStage.__init__(
            self,
            "chargediagram",
            data_settings,
            setpoint_settings,
            readout_methods,
            measurement_options=measurement_options,
            update_settings=update_settings,
            fit_options=fit_options,
        )

        self.segmented_ids: List[int] = []

        self.init_gate_values = []
        for gate in self.setpoint_settings['gates_to_sweep']:
            self.init_gate_values.append(gate.dc_voltage())

        self.range_change = range_change
        self.max_change = 0.5  # in volt
        self.min_change = 0.1  # in volt
        self.classifiers = classifiers

    def clean_up(self) -> None:
        """"""
        for ig, gate in enumerate(self.setpoint_settings['gates_to_sweep']):
            gate.dc_voltage(self.init_gate_values[ig])

    def check_quality(self) -> bool:
        """
        Checks quality of segments of a dataset. Returns true if any segment
        is classified as a good single or good double dot. The predicted
        category is saved in self.segment_ids:
        self.segment_ids =[(data_id, category), (..), .. ]

        Up to user to filter good dot regimes and check if it is the desired
        one.

        good single is category = 1
        good double is category = 3
        """
        assert self.segment_regimes is not None

        # Return true of there is some good regime among the segments
        found_categories = list(self.segment_regimes.values())
        if 1 in found_categories or 3 in found_categories:
            return True
        else:
            return False

    def determine_regime(
        self,
        segment_ids: List[int],
    ) -> Dict[int, int]:
        """"""
        seg_db_name = self.data_settings['segment_db_name']
        seg_db_folder = self.data_settings['segment_db_folder']

        with nt.switch_database(seg_db_name, seg_db_folder):
            segment_regimes = {}
            for data_id in segment_ids:
                goodsingle = self.classifiers['singledot'].predict(
                    data_id, seg_db_name, db_folder=seg_db_folder
                )
                gooddouble = self.classifiers['doubledot'].predict(
                    data_id, seg_db_name, db_folder=seg_db_folder
                )
                dotregime = self.classifiers['dotregime'].predict(
                    data_id, seg_db_name, db_folder=seg_db_folder
                )
                if any(goodsingle) and any(gooddouble):
                    # both good single and doubledot
                    # category = DOT_LABLE_MAPPING['doubledot'][1]
                    category = DOT_LABLE_MAPPING["singledot"][1]
                else:
                    # check if one is good and whether they contradict the
                    # dot regime prediction
                    if any(goodsingle) and not any(gooddouble):
                        # check if regime clf suggests a single dot as well
                        # dotregime = 0 => good single
                        if not any(dotregime):
                            category = DOT_LABLE_MAPPING["singledot"][1]
                        else:
                            category = DOT_LABLE_MAPPING["bothpoor"][0]

                    elif not any(goodsingle) and any(gooddouble):
                        if dotregime:
                            category = DOT_LABLE_MAPPING["doubledot"][1]
                        else:
                            category = DOT_LABLE_MAPPING["bothpoor"][0]

                    elif not any(goodsingle) and not any(gooddouble):
                        category = DOT_LABLE_MAPPING["bothpoor"][0]
                    else:
                        logger.error(
                            "ChargeDiagram.check_quality: Unable to "
                            + "assign dot quality. Unknown combination "
                            + "of single and doubledot predictions."
                        )

                segment_regimes[data_id] = category

                # save predicted category to metadata
                ds = load_by_id(data_id)
                nt_metadata = json.loads(ds.get_metadata(nt.meta_tag))
                nt_metadata["predicted_category"] = category
                ds.add_metadata(nt.meta_tag, json.dumps(nt_metadata))
        return segment_regimes

    def additional_post_measurement_actions(self) -> None:
        """"""
        db = self.data_settings['segment_db_name']
        db_loc = self.data_settings['segment_db_folder']
        segment_info = self.current_fit.save_segmented_data_return_info(db,
                                                                        db_loc)
        segment_info_new = []
        # convert to list of tuples which will hold their predicted label
        self.segmented_ids = [s_id for s_id in segment_info.keys()]
        logger.info("Derived run_ids: {}".format(self.segmented_ids))

        self.segment_regimes = self.determine_regime(self.segmented_ids)
        for seg_id, v_ranges in segment_info.items():
            segment_info_new.append((
                    seg_id,
                    v_ranges,
                    self.segment_regimes[seg_id],
            ))

        self.segment_info = segment_info_new

    @property
    def fit_class(self):
        """
        Use the appropriate fitting class
        """
        return DotFit

    def get_next_actions(self,
                         readout_method_to_use: Optional[str] = 'dc_current',
                         ) -> Tuple[List[str], List[str]]:
        """
        data fit returns next actions
        actions are:
        0: x more negative
        1: x more positive
        2: y more negative
        3: y more positive

        check if we reached safety limits and remove corresponding action
        if it is the case
        """
        all_actions = [
            "x more negative",
            "x more positive",
            "y more negative",
            "y more positive"
            ]
        issues = []
        actions = self.current_fit.next_actions[readout_method_to_use]

        for gid, gate in enumerate(self.setpoint_settings['gates_to_sweep']):
            d_n = abs(self.current_ranges[gid][0] - gate.safety_range()[0])
            d_p = abs(self.current_ranges[gid][1] - gate.safety_range()[1])

            if all_actions[gid * 2] in actions and d_n < 0.015:
                # gate reached negative limit
                issues.append(all_actions[gid * 2])
                actions.remove(all_actions[gid * 2])
            if all_actions[gid * 2 + 1] in actions and d_p < 0.015:
                # gate reached positive limit
                issues.append(all_actions[gid * 2 + 1])
                actions.remove(all_actions[gid * 2 + 1])
        return actions, issues

    def update_measurement_settings(
        self,
        actions: List[str],
    ) -> None:
        """
        Currently either shifting the window or making it larger.
        """
        all_actions = ["x more negative", "x more positive",
                        "y more negative", "y more positive"]
        for action in actions:
            if action not in all_actions:
                logger.error((f'{self.stage}: Unknown action.'
                    'Cannot update measurement setting'))

        if "x more negative" in actions:
            self._update_setpoint_settings(0, "negative")

        if "x more positive" in actions:
            self._update_setpoint_settings(0, "positive")

        if "y more negative" in actions:
            self._update_setpoint_settings(1, "negative")

        if "y more positive" in actions:
            self._update_setpoint_settings(1, "positive")
        else:
            logger.error((f'{self.stage}: Unknown action.'
                'Cannot update measurement setting'))

    def _update_setpoint_settings(self, gate_id: int, direction: str) -> None:
        curr_rng = self.current_ranges[gate_id]
        new_rng = list(curr_rng)
        gates_to_sweep = self.setpoint_settings['gates_to_sweep']
        max_rng = gates_to_sweep[gate_id].safety_range()

        if direction == "negative":
            if abs(curr_rng[0] - max_rng[0]) <= 2 * self.min_change:
                diff = abs(curr_rng[0] - max_rng[0])
            else:
                diff = self.range_change / 100 * abs(curr_rng[0] - max_rng[0])
                diff = min(diff, self.max_change)
                diff = max(diff, self.min_change)
            new_rng[0] = new_rng[0] - diff

        elif direction == "positive":
            if abs(curr_rng[1] - max_rng[1]) <= 2 * self.min_change:
                diff = abs(curr_rng[1] - max_rng[1])
            else:
                diff = self.range_change / 100 * abs(curr_rng[1] - max_rng[1])
                diff = min(diff, self.max_change)
                diff = max(diff, self.min_change)
            new_rng[1] = curr_rng[1] + diff
        else:
            raise NotImplementedError

        new_rng_tpl = tuple(new_rng)
        gates_to_sweep[gate_id].current_valid_range(new_rng_tpl)
        accepted_range = gates_to_sweep[gate_id].current_valid_range()
        self.current_ranges[gate_id] = accepted_range

        logger.info(f'Updating ranges of gate {gate_id} to {new_rng}')


