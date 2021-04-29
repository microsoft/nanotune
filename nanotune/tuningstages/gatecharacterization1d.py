from typing import Optional, Tuple, List, Union, Dict, Callable, Any, Sequence
import logging
import copy
import qcodes as qc

import nanotune as nt
from nanotune.device.gate import Gate
from nanotune.fit.pinchofffit import PinchoffFit
from nanotune.tuningstages.tuningstage import TuningStage
from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.classification.classifier import Classifier
from .base_tasks import ( # please update docstrings if import path changes
    check_measurement_quality,
    conclude_iteration_with_range_update,
    get_fit_range_update_directives,
)
from .gatecharacterization_tasks import (
    get_new_gatecharacterization_range,
    get_range_directives_gatecharacterization,
    set_gate_to_highest,
    finish_early_pinched_off,
)

logger = logging.getLogger(__name__)


class GateCharacterization1D(TuningStage):
    """Tuning stage performing individual gate characterizations.

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
        classifier: Pre-trained nt.Classifier predicting the quality of a
            pinchoff curve.
        noise_level: Relative level above which a measured output is considered
            being above the noise floor. Is compared to a normalized signal.
        main_readout_method: Readout method to use for early finish check.
        voltage_interval_to_track: Voltage interval over which the measured
            output is checked
        gate: The gate to sweep, an instance of nt.Gate.
        fit_class: PinchoffFit.
    """

    def __init__(
        self,
        data_settings: Dict[str, Any],
        setpoint_settings: Dict[str, Any],
        readout_methods: Dict[str, qc.Parameter],
        classifier: Classifier,
        noise_level: float = 0.001,  # compares to normalised signal
        main_readout_method: str = 'dc_current',
        voltage_interval_to_track = 0.3,
    ) -> None:
        """Initializes a gate characterization tuning stage.

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
            classifier: Pre-trained nt.Classifier predicting the quality of a
            pinchoff curve.
            noise_level: Relative level above which a measured output is considered
                being above the noise floor. Is compared to a normalized signal.
            main_readout_method: Readout method to use for early finish check.
            voltage_interval_to_track: Voltage interval over which the measured
                output is checked
        """

        TuningStage.__init__(
            self,
            "gatecharacterization1d",
            data_settings,
            setpoint_settings,
            readout_methods,
        )

        self.classifier = classifier
        self.noise_level = noise_level
        self.main_readout_method = main_readout_method
        self.voltage_interval_to_track = voltage_interval_to_track

        self._recent_readout_output: List[float] = []
        if isinstance(self.setpoint_settings['gates_to_sweep'], Gate):
            gate_list = [self.setpoint_settings['gates_to_sweep']]
            self.setpoint_settings['gates_to_sweep'] = gate_list

        assert len(self.setpoint_settings['gates_to_sweep']) == 1
        self.gate = self.setpoint_settings['gates_to_sweep'][0]

    @property
    def fit_class(self):
        """
        Data fitting class to extract pinchoff features.
        """
        return PinchoffFit

    def machine_learning_task(self, run_id) -> Dict[str, Any]:
        """Executes the post-measurement machine learning task, a binary
        classification predicting the quality of the measurement. The
        result is saved in a dictionary with 'quality' and 'regime' keys, the
        latter for completeness and compatibility with general tuning stage
        methods.

        Args:
            run_id: QCoDeS data run ID.

        Returns:
            dict: The classification outcome, saved under the 'quality' key. For
                completeness, the 'regime' key maps onto 'pinchoff'.
        """

        ml_result: Dict[str, Any] = {}
        ml_result['quality'] = check_measurement_quality(
            self.classifier,
            run_id,
            self.data_settings['db_name'],
            db_folder=self.data_settings['db_folder'],
        )
        ml_result['regime'] = 'pinchoff'
        return ml_result

    def verify_classification_result(
        self,
        ml_result: Dict[str, int],
    ) -> bool:
        """Verifies whether a good pinchoff curve has been predicted.

        Args:
            ml_result: Result returned by ``machine_learning_task``, a
                dictionary with 'quality' and 'regime' keys.

        Returns:
            bool: Whether the desired outcome has been found.
        """

        return bool(ml_result['quality'])

    def conclude_iteration(
        self,
        tuning_result: TuningResult,
        voltage_ranges: List[Tuple[float, float]],
        safety_voltage_ranges: List[Tuple[float, float]],
        current_iteration: int,
        max_n_iterations: int,
    ) -> Tuple[bool, List[Tuple[float, float]], List[str]]:
        """Method checking if one iteration of a run_stage measurement cycle has
        been successful. An iteration of such a measurement cycle takes data,
        performs a machine learning task, verifies and saves the machine
        learning result. If a repetition of this cycle is supported, then
        ``conclude_iteration`` determines whether another iteration should take
        place and which voltage ranges need to be measured.
        It wraps conclude_iteration_with_range_update in .base_tasks.py and
        resets self._recent_readout_output. self._recent_readout_output is used
        in self.finish_early to detect a pinched-off regime.

        Args:
            tuning_result: Result of the last run_stage measurement cycle.
            voltage_ranges: Voltage ranges last swept.
            safety_voltage_ranges: Safety voltage ranges, i.e. largest possible
                range that could be swept.
            current_iteration: Number of current iteration.
            max_n_iterations: Maximum number of iterations to perform before
                abandoning.

        Returns:
            bool: Whether this is the last iteration and the stage is done/to
                be stopped.
            list: New voltage ranges to sweep if the stage is not done.
            list: List of strings indicating any possible failure modes.
        """

        (done,
        new_voltage_ranges,
        termination_reasons) = conclude_iteration_with_range_update(
            tuning_result,
            voltage_ranges,
            safety_voltage_ranges,
            self.get_range_update_directives,
            get_new_gatecharacterization_range,
            count,
            max_n_iterations,
        )
        self._recent_readout_output = []
        return done, new_voltage_ranges, termination_reasons

    def get_range_update_directives(
        self,
        run_id: int,
        current_ranges: List[Tuple[float, float]],
        safety_ranges: List[Tuple[float, float]],
        ) -> Tuple[List[str], List[str]]:
        """Determines directives indicating if the current voltage ranges need
        to be extended or shifted. It first gets these directives from the data
        fit using get_fit_range_update_directives defined in .base_tasks.py and
        then checks if they can be put into action using
        get_range_directives_gatecharacterization defined in
        gatecharacterization_tasks.py. The check looks at whether safety ranges
        have been reached already, or whether a voltage range extension is
        possible.

        Args:
            run_id: QCoDeS data run ID.
            current_ranges: Last voltage range swept.
            safety_ranges: Safety range of the gate swept.

        """
        if isinstance(current_ranges, tuple):
            current_ranges = [current_ranges]
        if isinstance(safety_ranges, tuple):
            safety_ranges = [safety_ranges]

        fit_range_update_directives = get_fit_range_update_directives(
            self.fit_class,
            run_id,
            self.data_settings['db_name'],
            db_folder=self.data_settings['db_folder'],
        )
        (range_update_directives,
         issues) = get_range_directives_gatecharacterization(
            fit_range_update_directives,
            current_ranges,
            safety_ranges,
        )

        return range_update_directives, issues

    def finish_early(
        self,
        current_output_dict: Dict[str, float],
    ) -> bool:
        """Checks the average strength of measured signal over a given voltage
        interval is below the noise floor. If this is the case, the boolean
        returned indicates that the measurement can be stopped. It wraps
        ``finish_early_pinched_off`` defined in .gatecharacterization_tasks.py.

        Args:
            current_output_dict: Dictionary mapping strings indicating the
                readout method to QCoDeS parameters.

        Returns:
            bool: Whether the measurement can be stopped early.
        """

        param = self.readout_methods[self.main_readout_method]
        last_measurement_strength = current_output_dict[param.full_name]

        norm_consts = self.data_settings['normalization_constants']
        normalization_constant = norm_consts[self.main_readout_method]

        finish, self._recent_readout_output = finish_early_pinched_off(
            last_measurement_strength,
            normalization_constant,
            self._recent_readout_output,
            self.setpoint_settings['voltage_precision'],
            self.noise_level,
            self.voltage_interval_to_track,
        )

        return finish

