import logging
import json
import time
import datetime
import copy
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, List, Dict, Any, Sequence, Union, Generator
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
        self.current_setpoints: List[List[float]] = []
        self.result_ids: List[int] = []
        # self.current_id: int = -1
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

    @property
    @abstractmethod
    def fit_class(self):
        """"""
        pass

    @abstractmethod
    def get_range_update_directives(self) -> Tuple[List[str], List[str]]:
        """
        The fit is telling us what we need to to
        Return "false" as second argument if we should abandon the case
        """
        pass

    @abstractmethod
    def update_current_ranges(
    # def get_new_voltage_ranges(
        self,
        range_update_directives: List[str],
    ) -> List[List[float]]:
        """"""

    def save_features(
        self,
        current_id: int,
        ):
        """ """
        save_extracted_features(
            self.fit_class,
            current_id,
            self.data_settings['db_name'],
            db_folder=self.data_settings['db_folder'],
            fit_options=self.fit_options,
        )

    @abstractmethod
    def check_quality(self, run_id: int) -> bool:
        """"""

    @abstractmethod
    def predict_regime(self, run_id: int) -> bool:
        """"""

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

    def compute_setpoints(self) -> List[List[float]]:
        """
        """

        gates_to_sweep = self.setpoint_settings['gates_to_sweep']
        self.current_ranges = swap_range_limits_if_needed(
            gates_to_sweep,
            self.current_ranges,
        )
        setpoints = compute_linear_setpoints(
            self.current_ranges,
            self.setpoint_settings['voltage_precision'],
        )
        return setpoints

    @contextmanager
    def set_up_gates_for_measurement(self) -> Generator[None, None, None]:
        """ Ramp gates to start values before turning off ramping
        deactivate ramp - setpoints are calculated such that
        voltage differences do not exceed max_jump
        """
        for gg, gate in enumerate(self.setpoint_settings['gates_to_sweep']):
            gate.dc_voltage(self.current_setpoints[gg][0])
            gate.use_ramp(False)
            d = 0.01
            if self.measurement_options is not None:
                for read_method in self.readout_methods.keys():
                    options = self.measurement_options[read_method]
                    try:
                        d = max(d, float(options["delay"]))
                    except KeyError:
                        pass
            gate.post_delay(d)
        try:
            yield
        finally:
            for gate in self.setpoint_settings['gates_to_sweep']:
                gate.use_ramp(True)
                gate.post_delay(0)

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

    def _run_stage(
        self,
        plot_measurements: bool = True,
    ) -> Tuple[bool, List[str]]:
        """"""
        done = False
        count = 0
        termination_reasons: List[str] = []
        range_update_directives: List[str] = []

        nt.set_database(
            self.data_settings['db_name'],
            db_folder=self.data_settings['db_folder']
        )

        while not done:
            count += 1
            logger.info("Iteration no " + str(count))

            self.current_setpoints = self.compute_setpoints()
            current_id = self.measure(self.current_setpoints)

            self.result_ids.append(current_id)
            self.save_features(current_id)
            self.additional_post_measurement_actions()

            measurement_quality = self.check_quality(current_id)
            save_classification_result(
                current_id,
                "predicted_quality",
                measurement_quality,
            )
            predicted_regime = self.predict_regime(current_id)
            save_classification_result(
                current_id,
                "predicted_regime",
                predicted_regime,
            )

            if plot_measurements:
                plot_fit(
                    self.fit_class,
                    current_id,
                    self.data_settings['db_name'],
                    db_folder=self.data_settings['db_folder'],
                )

            success = bool(measurement_quality)
            if success:
                done = True
                termination_reasons = []
            elif self.update_settings:
                (range_update_directives,
                 termination_reasons) = self.get_range_update_directives()

                if not range_update_directives:
                    done = True
                    success = False
                else:
                    self.update_current_ranges(range_update_directives)
                    done = False
                    success = False

            if count >= self.max_count:
                done = True
                success = False
                termination_reasons.append("max count reached")

            print_tuningstage_status(
                self.stage,
                success,
                predicted_regime,
                range_update_directives,
                termination_reasons,
            )

        self.clean_up()
        return success, termination_reasons

    def run_stage(
        self,
        plot_measurements: bool = True,
    ) -> TuningResult:
        """
        """
        success, termination_reasons = self._run_stage(
            plot_measurements=plot_measurements
        )

        tuning_result = TuningResult(
            self.stage,
            success,
            termination_reasons=termination_reasons,
            data_ids=self.result_ids,
            db_name=self.data_settings['db_name'],
            db_folder=self.data_settings['db_folder'],
            features=get_measurement_features(
                self.result_ids[-1],
                self.data_settings['db_name'],
                db_folder=self.data_settings['db_folder'],
            ),
            timestamp=datetime.datetime.now().isoformat(),
        )

        return tuning_result
