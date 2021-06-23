# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Any
import qcodes as qc
import nanotune as nt
from nanotune.classification.classifier import Classifier
from nanotune.device.device import NormalizationConstants

@dataclass
class DataSettings:
    db_name: str = nt.config['db_name']
    db_folder: str = nt.config['db_folder']
    normalizations_constants: Optional[NormalizationConstants] = None
    experiment_id: Optional[int] = None
    segment_db_name: Optional[str] = None
    segment_db_folder: Optional[str] = None
    segment_experiment_id: Optional[int] = None


@dataclass
class SetpointSettings:
    voltage_precision: float
    parameters_to_sweep: Optional[Sequence[qc.Parameter]] = None
    ranges_to_sweep: Optional[Sequence[Sequence[float]]] = None
    safety_voltage_ranges: Optional[Sequence[Sequence[float]]] = None
    setpoint_method: Optional[
        Callable[[Any], Sequence[Sequence[float]]]] = None


@dataclass
class Classifiers:
    pinchoff: Optional[Classifier] = None
    singledot: Optional[Classifier] = None
    doubledot: Optional[Classifier] = None
    dotregime: Optional[Classifier] = None