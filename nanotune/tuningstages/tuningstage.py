import logging
import json
import time
import copy
import numpy as np
from functools import partial
from abc import ABCMeta, abstractmethod
from typing import (
    Optional, Tuple, List, Dict, Any, Sequence, Union, Generator, Callable,
)
from contextlib import contextmanager

import qcodes as qc
from qcodes.dataset.measurements import Measurement as QC_Measurement
from qcodes.dataset.experiment_container import load_by_id
from qcodes.instrument.visa import VisaInstrument
from nanotune.device_tuner.tuningresult import TuningResult
import nanotune as nt
from nanotune.device.gate import Gate
from .take_data import take_data, ramp_to_setpoint
from .base_tasks import (
    save_classification_result,
    save_extracted_features,
    set_up_gates_for_measurement,
    prepare_metadata,
    save_metadata,
    compute_linear_setpoints,
    swap_range_limits_if_needed,
    get_elapsed_time,
    plot_fit,
    get_measurement_features,
    take_data_add_metadata,
    print_tuningstage_status,
    run_stage,
    iterate_stage,
)
logger = logging.getLogger(__name__)


class TuningStage(metaclass=ABCMeta):
    """Base class implementing the common sequence of a tuning stage.

    Attributes:
        stage: String identifier indicating which stage it implements, e.g.
            gatecharacterization.
        data_settings: Dictionary with information about data, e.g. where it
            should be saved and how it should be normalized.
            Required fields are 'db_name', 'db_folder' and
            'normalization_constants'.
        setpoint_settings: Dictionary with information about how to compute
            setpoints. Required fields are 'gates_to_sweep' and
            'voltage_precision'.
        readout_methods: Dictionary mapping string identifiers such as
            'dc_current' to QCoDeS parameters measuring/returning the desired
            quantity (e.g. current throught the device).
        current_ranges: List of voltages ranges (tuples of floats) to measure.
        safety_ranges: List of satefy voltages ranges, i.e. safety limits within
            which gates don't blow up.
        fit_class: Abstract property, to be specified in child classes. It is
            the class that should perform the data fitting, e.g. PinchoffFit.
    """

    def __init__(
        self,
        stage: str,
        data_settings: Dict[str, Any],
        setpoint_settings: Dict[str, Any],
        readout_methods: Dict[str, qc.Parameter],
    ) -> None:
        """Initializes the base class of a tuning stage. Voltages to sweep and
        safety voltages are determined from the list of gates in
        setpoint_settings.

        Args:
            stage: String identifier indicating which stage it implements, e.g.
                gatecharacterization.
            data_settings: Dictionary with information about data, e.g. where it
                should be saved and how it should be normalized.
                Required fields are 'db_name', 'db_folder' and
                'normalization_constants'.
            setpoint_settings: Dictionary with information about how to compute
                setpoints. Fie
            readout_methods: Dictionary mapping string identifiers such as
                'dc_current' to QCoDeS parameters measuring/returning the
                desired quantity (e.g. current throught the device).
        """

        self.stage = stage
        self.data_settings = data_settings
        self.setpoint_settings = setpoint_settings
        self.readout_methods = readout_methods

        self.current_ranges: List[Tuple[float, float]] = []
        self.safety_ranges: List[Tuple[float, float]] = []

        for gate in self.setpoint_settings['gates_to_sweep']:
            curr_rng = gate.current_valid_range()
            sfty_rng = gate.safety_range()
            if not curr_rng:
                logger.warning(
                    "No current valid ranges for " + gate.name + ". "
                    + "Taking safety range."
                )
                if isinstance(sfty_rng, list):
                    sfty_rng = tuple(sfty_rng)
                curr_rng = sfty_rng

            self.current_ranges.append(curr_rng)
            self.safety_ranges.append(sfty_rng)

    @property
    @abstractmethod
    def fit_class(self):
        """To be specified in child classes. It is  """
        pass

    @abstractmethod
    def conclude_iteration(
        self,
        tuning_result: TuningResult,
        voltage_ranges: List[Tuple[float, float]],
        safety_voltage_ranges: List[Tuple[float, float]],
        count: int,
        n_iterations: int,
    ) -> Tuple[bool, List[Tuple[float, float]], List[str]]:
        """ """
        pass

    @abstractmethod
    def verify_classification_result(
        self,
        ml_result: Dict[str, int],
    ) -> bool:
        """ """
        pass

    @abstractmethod
    def machine_learning_task(
        self,
        run_id: int,
    ) -> Dict[str, Any]:
        """ """
        pass

    def save_machine_learning_result(
        self,
        current_id: int,
        ml_result: Dict[str, int],
        ):
        """ """

        save_extracted_features(
            self.fit_class,
            current_id,
            self.data_settings['db_name'],
            db_folder=self.data_settings['db_folder'],
        )
        for result_type, result_value in ml_result.items():
            save_classification_result(
                current_id,
                result_type,
                result_value,
            )

    def clean_up(self) -> None:
        """"""
        pass

    def finish_early(
        self,
        current_output_dict: Dict[str, float],
    ) -> bool:
        """"""
        return False

    def compute_setpoints(
        self,
        voltage_ranges: List[Tuple[float, float]],
    ) -> List[List[float]]:
        """
        """

        setpoints = compute_linear_setpoints(
            voltage_ranges,
            self.setpoint_settings['voltage_precision'],
        )
        return setpoints

    def show_result(
        self,
        plot_result: bool,
        current_id: int,
        tuning_result: TuningResult,
    ) -> None:
        """ """
        if plot_result:
            plot_fit(
                self.fit_class,
                current_id,
                self.data_settings['db_name'],
                db_folder=self.data_settings['db_folder'],
            )
        print_tuningstage_status(tuning_result)

    def prepare_nt_metadata(self) -> Dict[str, Any]:
        """ """
        nt_meta = prepare_metadata(
            self.setpoint_settings['gates_to_sweep'][0].parent.name,
            self.data_settings['normalization_constants'],
            self.readout_methods
        )
        return nt_meta

    def measure(
        self,
        setpoints: List[List[float]],
    ) -> int:
        """
        """

        run_id = take_data_add_metadata(
            self.setpoint_settings['gates_to_sweep'],
            list(self.readout_methods.values()),
            setpoints,
            finish_early_check=self.finish_early,
            do_at_inner_setpoint=ramp_to_setpoint,
            pre_measurement_metadata=self.prepare_nt_metadata()
        )

        return run_id

    def run_stage(
        self,
        iterate: bool = True,
        max_iterations: int = 10,
        plot_result: bool = True,
    ) -> TuningResult:
        """"""

        nt.set_database(
            self.data_settings['db_name'],
            db_folder=self.data_settings['db_folder']
        )

        self.current_ranges = swap_range_limits_if_needed(
            self.setpoint_settings['gates_to_sweep'],
            self.current_ranges,
        )

        run_stage_tasks = [
            self.compute_setpoints,
            self.measure,
            self.machine_learning_task,
            self.save_machine_learning_result,
            self.verify_classification_result,
        ]
        if not iterate:
            max_iterations = 1

        tuning_result = iterate_stage(
            self.stage,
            self.current_ranges,
            self.safety_ranges,
            run_stage,
            run_stage_tasks,  # type: ignore
            self.conclude_iteration,
            partial(self.show_result, plot_result),
            max_iterations,
        )

        tuning_result.db_name = self.data_settings['db_name']
        tuning_result.db_folder = self.data_settings['db_folder']

        self.clean_up()
        return tuning_result

