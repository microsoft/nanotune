from __future__ import annotations
import os
import re
import logging
import time
import datetime
import json
import copy
from typing import List, Optional, Dict, Tuple, Sequence, Callable, Any, Union

import qcodes as qc
from qcodes import validators as vals
import nanotune as nt
from nanotune.device_tuner.tuningreport import TuningReport
logger = logging.getLogger(__name__)


class TuningResult():
    def __init__(
        self,
        device_name: str,
    ) -> None:
        self.device_name = device_name
        self._data_comments: Dict[int, str] = {}
        self._data_comments["general"] = ""
        self._stage_summaries: Dict[str, Any] = {}

    @property
    def data_comments(self):
        return self._data_comments

    @data_comments.setter
    def data_comments(self, new_comments: Dict[int, str]):
        return self.update_data_comments(new_comments)

    @property
    def stage_summaries(self):
        return self._stage_summaries

    @stage_summaries.setter
    def stage_summaries(self, new_summary: Dict[str, Any]):
        return self.update_stage_summaries(new_summary)

    def add_result(
        self,
        identifier: str,
        success: bool,
        termination_reasons: List[int],
        result: Dict[str, Any],
        comment: Optional[str],
        ) -> None:
        """ """
        summary = {}
        summary[identifier] = {
            'success': success,
            'termination_reasons': termination_reasons,
            **result,
            'comment': comment,
        }
        self.update_stage_summaries(summary)

    def update(self,
        tuning_result: TuningResult,
        ) -> None:
        """ """
        if tuning_result.device_name() != self.device_name():
            raise ValueError('Device names do not match.')
        self.update_data_comments(tuning_result.data_comments)
        self.update_stage_summaries(tuning_result.stage_summaries)

    def update_data_comments(self,
        data_comments: Dict[int, str],
        ) -> None:
        """ """
        common_keys = list(
            set(data_comments.keys()) & set(self._data_comments.keys())
            )
        for key in common_keys:
            data_comments[key] += (';' + self._data_comments[key])

        self.data_comments.update(data_comments)

    def update_stage_summaries(self, stage_summaries: Dict[str, Any]) -> None:
        """ Checks if keys in stage_summaries overlap with already exiting
        keys.
        """
        common_keys = list(
            set(stage_summaries.keys()) & set(self._stage_summaries.keys())
            )
        for key in common_keys:
            if stage_summaries[key] != self._stage_summaries[key]:
                if key[-1].isdigit():
                    m = re.search(r'\d+$', key).group()
                    key[:-len(m)] + str(int(m)+1)
                else:
                    new_key = key + '_2'
                stage_summaries[new_key] = stage_summaries[key]
                del stage_summaries[key]
        self._stage_summaries.update(stage_summaries)

    def get_stage_summaries(self) -> Dict[str, Any]:
        """"""
        return copy.deepcopy(self._stage_summaries)

    def get_data_comments(self) -> Dict[str, Any]:
        """"""
        return copy.deepcopy(self._data_comments)

    def remove_data_comments(self,
        data_ids: List[int],
        ) -> None:
        """
        """
        for data_id in data_ids:
            del self._data_comments[data_id]

    def remove_stage_summary(self,
        stage: str,
        ) -> None:
        """ """
        del self._stage_summaries[stage]

    def print(
        self,
        tuning_step: str,
        send: bool = False,
    ) -> None:
        """"""
        nl = TuningReport(
            self.stage_summaries,
            device_name=self.device_name(),
            comments=self.data_comments,
        )
        nl.press()
        if send:
            nl.distribute(reciever=["me@onenote.com", "nanotune@outlook.com"],
                          sender="nanotune@outlook.com",
                          pwd="N@notune")



