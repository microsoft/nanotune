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
SETPOINT_METHODS = nt.config["core"]["setpoint_methods"]
data_dimensions = {
    'gatecharacterization1d': 1,
    'chargediagram': 2,
    'coulomboscillations': 1,
}


class TuningStage(metaclass=ABCMeta):
    """
    readout_methods = {'dc_current': qc.Parameter,
                    'dc_sensor': qc.Parameter}
    }
    setpoint_settings = {
        'voltage_precision':
        'gates_to_sweep':
    }
    measurement_options = {
        'dc_current': {'delay': 0.1,
                        'inter_delay': 0.1,
    }
    }
    data_settings = {
        'db_name': '',
        'normalization_constants': {},
        'db_folder': '',
    }
    fit_options = {},
    """
    def __init__(
        self,
        stage: str,
        data_settings: Dict[str, Any],
        setpoint_settings: Dict[str, Any],
        readout_methods: Dict[str, qc.Parameter],
        measurement_options: Optional[Dict[str, Dict[str, Any]]] = None,
        update_settings: bool = True,
        fit_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._D = data_dimensions[stage]
        self.data_settings = data_settings
        self.setpoint_settings = setpoint_settings
        self.readout_methods = readout_methods
        self.measurement_options = measurement_options
        if fit_options is None:
            fit_options = {}
        self.fit_options = fit_options

        self.stage = stage

        self.update_settings = update_settings

        self.current_ranges: List[Tuple[float, float]] = []
        self.safety_ranges: List[Tuple[float, float]] = []
        self.current_setpoints: List[List[float]] = []
        self.max_count = 10

        for gate in self.setpoint_settings['gates_to_sweep']:
            if not gate.current_valid_range():
                logger.warning(
                    "No current valid ranges for "
                    + gate.name
                    + " given. Taking entire range."
                )
                curr_rng = gate.safety_range()

            else:
                # sweep to max ranges if current valid range is close to save
                # us a potential second 2D sweep
                curr_rng = np.array(gate.current_valid_range())
                sfty_rng = np.array(gate.safety_range())

                close = np.isclose(curr_rng, sfty_rng, 0.05)
                for idx, isclose in enumerate(close):
                    if isclose:
                        curr_rng[idx] = sfty_rng[idx]
                if isinstance(curr_rng, np.ndarray):
                    curr_rng = curr_rng.tolist()

            gate.current_valid_range(curr_rng)
            self.current_ranges.append(curr_rng)
            self.safety_ranges.append(gate.safety_range())

    @property
    @abstractmethod
    def fit_class(self):
        """"""
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

    @abstractmethod
    def verify_classification_result(
        self,
        ml_result: Dict[str, int],
    ) -> bool:
        """ """

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

    @abstractmethod
    def machine_learning_task(self, run_id) -> Dict[str, Any]:
        """ """

    def clean_up(self) -> None:
        """"""
        pass

    def additional_post_measurement_actions(self) -> None:
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
        plot_result: Optional[bool] = True,
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

        tuning_result = iterate_stage(
            self.stage,
            self.current_ranges,
            self.safety_ranges,
            run_stage,
            run_stage_tasks,  # type: ignore
            self.conclude_iteration,
            partial(self.show_result, plot_result),
            self.max_count,
        )

        tuning_result.db_name = self.data_settings['db_name']
        tuning_result.db_folder = self.data_settings['db_folder']

        self.clean_up()
        return tuning_result

