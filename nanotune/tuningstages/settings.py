# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass, field
from typing import Optional, Sequence, Callable, Any, Union, Dict
import qcodes as qc
import nanotune as nt
from nanotune.classification.classifier import Classifier
from nanotune.device.device import NormalizationConstants


class Settings:
    """Base class for settings dataclasses such as `DataSettings` or
    `SetpointSettings`.
    """
    def update(
        self,
        new_settings: Union[
            Dict[str, Sequence[Any]], Settings],
    ) -> None:
        """Updates attributes with new settings. Raises a ValueError if an
        attribute is not found or input argument is neither a Dict, nor a
        `Setting`.

        Args:
            new_settings (Dict or Settings):
        """
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
    """`Setting` sub-class holding data-related information such as where
    data is saved or how it is normalized.

    Attributes:
        db_name (str): database name. Default set to `nt.config['db_name']`.
        db_folder (str): path of folder containing `db_name`. Default set to
            `nt.config['db_folder']`.
        normalization_constants (NormalizationConstants): device specific
            normalization constants.
        experiment_id (int): ID of experiment to which tuning data belongs,
            optional.
        segment_db_name (str): name of database containing segmented dot data,
            saved when performing a dot fit.
        segment_db_folder (str): path of folder containing `segment_db_name`.
            Default set to `nt.config['db_folder']`.
        segment_experiment_id (int): ID of experiment to which dot segment data
            belongs, optional.
        segment_size (float): voltage range/span of each dot segment, classified
            independently.
        noise_floor (float): threshold below which a measured signal is
            considered noise. Compared to normalized measurements.
        dot_signal_threshold (float): threshold below which a measured signal is
            considered possibly belong to a few-electron regime and above which
            a signal is considered open current.
    """
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
        """Updates attributed with new settings by calling `super`'s update
        method and then overwriting `normalization_constants` with and instance
        of `NormalizationConstants`.

        Args:
            new_settings (Dict or Settings):
        """
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
        if isinstance(new_settings, DataSettings):
            self.normalization_constants = NormalizationConstants(
                    **asdict(new_settings.normalization_constants)
                )


@dataclass
class SetpointSettings(Settings):
    """`Settings` sub-class holding setpoint-related information such as
    voltage precision and parameters to sweep.

    Attributes:
        voltage_precision (float): voltage difference between setpoints.
        parameters_to_sweep (Sequence[qc.Parameter]): list of QCoDeS parameters
            to sweep.
        ranges_to_sweep (Sequence[Sequence[float]]): voltage ranges to sweep,
            in same order as `parameters_to_sweep`.
        safety_voltage_ranges (Sequence[Sequence[float]]): safe voltage ranges
            of `parameters_to_sweep`, in the same order.
        setpoint_method (optional Callable): optional callable, to be used to
            calculate setpoints. Default are linearly spaced setpoints.
    """
    voltage_precision: float
    parameters_to_sweep: Sequence[qc.Parameter] = field(default_factory=list)
    ranges_to_sweep: Sequence[Sequence[float]] = field(default_factory=list)
    safety_voltage_ranges: Sequence[Sequence[float]] = field(default_factory=list)
    setpoint_method: Optional[
        Callable[[Any], Sequence[Sequence[float]]]] = None


@dataclass
class Classifiers:
    """Class grouping binary classifiers required for tuning.

    Attributes:
        pinchoff (optional nt.Classifier): pre-trained pinch-off classifier.
        singledot (optional nt.Classifier): pre-trained single dot classifier.
        doubledot (optional nt.Classifier): pre-trained double dot classifier.
        dotregime (optional nt.Classifier): pre-trained dot regime classifier.
    """
    pinchoff: Optional[Classifier] = None
    singledot: Optional[Classifier] = None
    doubledot: Optional[Classifier] = None
    dotregime: Optional[Classifier] = None

    def is_dot_classifier(self) -> bool:
        """Checks if dot classifiers are specified/not None. If so, the
        `Classifiers` instance can be used in a dot tuning algorithm.
        """
        if (self.singledot is not None and
            self.doubledot is not None and self.dotregime is not None):
            return True
        else:
            return False

    def is_pinchoff_classifier(self) -> bool:
        """Checks if pinch-off classifier is specified/not None. If so, the
        `Classifiers` instance can be used gate or device characterizations.
        """
        if self.pinchoff is not None:
            return True
        else:
            return False
