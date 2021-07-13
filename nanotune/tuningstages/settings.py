# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass, field
from typing import Optional, Sequence, Callable, Any, Union, Dict, Tuple, List
import qcodes as qc
import nanotune as nt
from nanotune.classification.classifier import Classifier
from nanotune.device.device import NormalizationConstants


class Settings:
    def update(
        self,
        new_settings: Union[
            Dict[str, Sequence[Any]], Settings],
    ) -> None:
        if is_dataclass(new_settings):
            new_constants_dict = asdict(new_settings)
        elif isinstance(new_settings, Dict):
            new_constants_dict = new_settings
        else:
            raise ValueError('Invalid settings. Use the appropriate \
                 dataclass or a Dict instead.')

        for sett_type, setting in new_constants_dict.items():
            if not hasattr(self, sett_type):
                raise KeyError(f'Invalid setting subfield.')
            setattr(self, sett_type, setting)


@dataclass
class DataSettings(Settings):
    db_name: str = nt.config['db_name']
    db_folder: str = nt.config['db_folder']
    normalization_constants: NormalizationConstants = NormalizationConstants()
    experiment_id: Optional[int] = None
    segment_db_name: str = f'segmented_{nt.config["db_name"]}'
    segment_db_folder: str = nt.config['db_folder']
    segment_experiment_id: Optional[int] = None
    segment_size: float = 0.05
    noise_floor: float = 0.02
    dot_signal_threshold: float = 0.1

    def update(
        self,
        new_settings: Union[
            Dict[str, Sequence[Any]], Settings],
    ) -> None:
        super().update(new_settings)
        if isinstance(new_settings, Dict):
            if 'normalization_constants' in new_settings.keys():
                constants = new_settings['normalization_constants']
                if isinstance(constants, Dict):
                    self.normalization_constants = NormalizationConstants(
                        **constants
                    )
                if is_dataclass(constants):
                    self.normalization_constants = NormalizationConstants(
                        **asdict(constants)
                    )
        if is_dataclass(new_settings):
            self.normalization_constants = NormalizationConstants(
                    **asdict(new_settings.normalization_constants)
                )


@dataclass
class SetpointSettings(Settings):
    voltage_precision: float
    parameters_to_sweep: Sequence[qc.Parameter] = field(default_factory=list)
    ranges_to_sweep: Sequence[Sequence[float]] = field(default_factory=list)
    safety_voltage_ranges: Sequence[Sequence[float]] = field(default_factory=list)
    setpoint_method: Optional[
        Callable[[Any], Sequence[Sequence[float]]]] = None


@dataclass
class Classifiers:
    pinchoff: Optional[Classifier] = None
    singledot: Optional[Classifier] = None
    doubledot: Optional[Classifier] = None
    dotregime: Optional[Classifier] = None

    def is_dot_classifier(self) -> bool:
        if (self.singledot is not None and
            self.doubledot is not None and self.dotregime is not None):
            return True
        else:
            return False

    def is_pinchoff_classifier(self) -> bool:
        if self.pinchoff is not None:
            return True
        else:
            return False
