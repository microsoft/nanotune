from __future__ import annotations
import os
import re
import logging
import time
import datetime
import json
import copy
from typing import List, Optional, Dict, Tuple, Sequence, Callable, Any, Union
from dataclasses import dataclass, asdict, field

import qcodes as qc
from qcodes import validators as vals
import nanotune as nt

logger = logging.getLogger(__name__)


@dataclass()
class TuningResult:
    """Container to hold tuning result data

    Attributes:
        stage (str): the tuning stage, e.g. 'chargediagram'. Can be any string,
            not necessarily a nt.tuningstage.
        success (bool): whether the stage terminated successfully
        guids (list): List of GUIDs of the measurements taken during the
            stage
        termination_reasons (list): List of reasons why the stage did not
            succeed. Example: 'no current'.
        data_ids (list): List of qc.run_id of the the measurements taken during
            the stage. Optional, for convenience.
        db_name (str): The database where to find data_ids, optional.
        db_folder (str): The database folder where to find db_name and thus
            data_ids. Optional
        comment (str): Optional string if there is anything to say about the
            tuning.
        timestamp (str): time stamp when the stage finished.

    Methods:
        to_dict: Export tuning results to dictionary
        to_json: Export tuning results to JSON
    """

    stage: str
    success: bool
    guids: List[str] = field(default_factory=list)
    ml_result: Dict[str, Any] = field(default_factory=dict)
    data_ids: List[str] = field(default_factory=list)
    db_name: str = ""
    db_folder: str = ""
    termination_reasons: List[str] = field(default_factory=list)
    comment: str = ""
    timestamp: str = ""
    status: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns:
            dict: TuningResult instance as dict
        """

        return asdict(self)

    def to_json(self) -> str:
        """
        Returns:
            str: serialized version TuningResult instance
        """

        return json.dumps(self.to_dict())


class MeasurementHistory:
    """Container to save tuning results. Each device should have its own
    instance. Results are saved in a dictionary mapping string indentifiers to
    TuningResult instances.

    Attributes:
        device_name (str): Name of device tuned.
        tuningresults (dict): Dictionary mapping string identifiers to
            instances of TuningResults.

    Methods:
        add_tuningresult(new_tuningresult, identifier=None): Add another
            result. If no identifier is specified,
            new_tuningresult.name + '_' + new_tuningresult.guids[-1] is used.
        to_dict: Export all tuning results to a dictionary.
        to_json: Export all tuning results to JSON formatted string.
    """

    def __init__(
        self,
        device_name: str,
    ) -> None:
        """
        Args:
            device_name (str): Name of device tuned.
        """

        self.device_name = device_name
        self._tuningresults: Dict[str, TuningResult] = {}

    @property
    def tuningresults(self):
        """tuningresults property getter """
        return self._tuningresults

    @tuningresults.setter
    def tuningresults(
        self,
        new_tuningresult: Union[TuningResult, Dict[str, TuningResult]],
    ) -> None:
        """tuningresults property setter. Checks if keys in new_tuningresult
        already exist in self.tuningresults.
        If a bare TuningResult instance is given, the TuningResult.stage serves
        as identifier.

        Args:
            new_tuningresult (Union[TuningResult, Dict[str, TuningResult]]):

        Returns:
        """
        if isinstance(new_tuningresult, TuningResult):
            new_tuningresult = {new_tuningresult.stage: new_tuningresult}

        common_keys = list(
            set(new_tuningresult.keys()) & set(self._tuningresults.keys())
        )
        for key in common_keys:
            if new_tuningresult[key] != self._tuningresults[key]:
                try:
                    append_idx = new_tuningresult[key].guids[-1]
                except IndexError:
                    append_idx = new_tuningresult[key].data_ids[-1]
                new_key = key + f"_{append_idx}"
                new_tuningresult[new_key] = new_tuningresult[key]
                del new_tuningresult[key]

        self._tuningresults.update(new_tuningresult)

    def to_dict(self) -> Dict[str, Any]:
        """Merges all MeasurementHistory attributed into a dict

        Returns:
            dict: all MeasurementHistory data in a dict
        """

        self_dict = {k: v.to_dict() for (k, v) in self.tuningresults.items()}
        self_dict["device_name"] = self.device_name
        return self_dict

    def to_json(self) -> str:
        """Serializes MeasurementHistory instance to a JSON formatted sting.

        Returns:
            str: JSON formatted serial version of MeasurementHistory instance
        """

        return json.dumps(self.to_dict())

    def add_result(
        self,
        tuningresult: TuningResult,
        identifier: Optional[str] = None,
    ) -> None:
        """Adds TuningResult instance to self.tuningresults dictionary.

        Args:
            tuningresult (TuningResult):
            identifier (Optional[str]): default if not supplied is
                tuningresult.stage
        """

        if identifier is None:
            identifier = tuningresult.stage
        self.tuningresults = {identifier: tuningresult}

    def update(self, other_measurement_history: MeasurementHistory) -> None:
        """ """
        assert other_measurement_history.device_name == self.device_name

        # not using dict.merge in case of key duplicates - which are taken
        # care of in add_result
        for key, result in other_measurement_history.tuningresults.items():
            self.add_result(result, key)
