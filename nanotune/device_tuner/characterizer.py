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
from nanotune.classification.classifier import Classifier
from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.device.gate import Gate
from nanotune.device_tuner.tuner import Tuner, set_back_voltages

logger = logging.getLogger(__name__)


class Characterizer(Tuner):
    """
    classifiers = {
        'pinchoff': Optional[Classifier],
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
    measurement_options = {
        'delay': float,
        'inter_delay': float,
        'setpoint_method': str,
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
        super().__init__(name,
            data_settings,
            classifiers,
            setpoint_settings,
            fit_options=fit_options)


    def characterize(self,
        device: Nt_Device,
        gate_configurations: Optional[Dict[int, Dict[int, float]]] = None,
        ) -> TuningResult:
        """
        gate_configurations: Dict[int, Dict[int, float]]; with gate
        configuration to be applied for individual gate characterizations.
        Example: Set top barrier of a 2DEG device.
        """
        if self.qcodes_experiment.sample_name != Nt_Device.name:
            logger.warning(
                ("The device's name does match the"
                 " the sample name in qcodes experiment.")
            )
        if gate_configurations is None:
            gate_configurations = {}

        tuningresult = TuningResult(
            'device_characterization_result', device.name
            )

        for gate in device.gates:
            with set_back_voltages(device.gates):
                gate_id = gate.layout_id()
                if gate_id in gate_configurations.keys():
                    for other_id, dc_voltage in gate_configurations[gate_id]:
                        device.gates[other_id].dc_voltage(dc_voltage)

                sub_result = self.characterize_gates(
                    device, gates=device.gates,
                    use_safety_ranges=True,
                    )
                tuningresult.update(sub_result)

        if device.name not in self.tuningresults.keys():
            self.tuningresults[device.name] = {}
        self.tuningresults[device.name].update(tuningresult)

        return tuningresult













