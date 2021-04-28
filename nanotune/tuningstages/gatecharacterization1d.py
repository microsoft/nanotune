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
from .base_tasks import (
    check_measurement_quality,
    conclude_iteration_with_range_update,
    get_fit_range_update_directives,
)
from .gatecharacterization_tasks import (
    get_new_gatecharacterization_range,
    get_range_directives_gatecharacterization,
)

logger = logging.getLogger(__name__)


class GateCharacterization1D(TuningStage):
    """
    """

    def __init__(
        self,
        data_settings: Dict[str, Any],
        setpoint_settings: Dict[str, Any],
        readout_methods: Dict[str, qc.Parameter],
        classifier: Classifier,
        fit_options: Optional[Dict[str, Any]] = None,
        noise_level: float = 0.001,  # compares to normalised signal
    ) -> None:
        """
        """
        TuningStage.__init__(
            self,
            "gatecharacterization1d",
            data_settings,
            setpoint_settings,
            readout_methods,
            fit_options=fit_options,
        )

        self.classifier = classifier
        self.noise_level = noise_level
        self._recent_signals: List[float] = []
        if isinstance(self.setpoint_settings['gates_to_sweep'], Gate):
            gate_list = [self.setpoint_settings['gates_to_sweep']]
            self.setpoint_settings['gates_to_sweep'] = gate_list
        else:
            assert len(self.setpoint_settings['gates_to_sweep']) == 1
        self.gate = self.setpoint_settings['gates_to_sweep'][0]

    @property
    def fit_class(self):
        """
        Use the appropriate fitting class
        """
        return PinchoffFit

    def machine_learning_task(self, run_id) -> Dict[str, Any]:
        """ """
        ml_result: Dict[str, Any] = {}
        ml_result['quality'] = self.check_quality(run_id)
        ml_result['regime'] = 'pinchoff'
        return ml_result

    def check_quality(self, run_id: int) -> bool:
        """"""
        qual = check_measurement_quality(
            self.classifier,
            run_id,
            self.data_settings['db_name'],
            db_folder=self.data_settings['db_folder'],
        )
        return qual


    def conclude_iteration(
        self,
        tuning_result: TuningResult,
        voltage_ranges: List[Tuple[float, float]],
        safety_voltage_ranges: List[Tuple[float, float]],
        count: int,
        max_n_iterations: int,
    ) -> Tuple[bool, List[Tuple[float, float]], List[str]]:
        """ """

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
        return done, new_voltage_ranges, termination_reasons


    def verify_classification_result(
        self,
        ml_result: Dict[str, int],
    ) -> bool:
        """ """
        return bool(ml_result['quality'])


    # def get_new_current_ranges(
    #     self,
    #     current_ranges: List[Tuple[float, float]],
    #     safety_ranges: List[Tuple[float, float]],
    #     range_update_directives: List[str],
    # ) -> List[Tuple[float, float]]:
    #     """"""

    #     new_current_ranges = get_new_gatecharacterization_range(
    #         current_ranges,
    #         safety_ranges,
    #         range_update_directives,
    #     )
    #     return new_current_ranges

    def get_range_update_directives(
        self,
        current_id: int,
        current_ranges: List[Tuple[float, float]],
        safety_ranges: List[Tuple[float, float]],
        ) -> Tuple[List[str], List[str]]:
        """
        Define range_update_directives if quality of current fit is sub-optimal.
        First: try to sweep the current gate more negative.
        If we are speeing to its min_v, we set the auxiliary_gate to min_v
        No intermediate tries, just being efficient.
        """
        fit_range_update_directives = get_fit_range_update_directives(
            self.fit_class,
            current_id,
            self.data_settings['db_name'],
            db_folder=self.data_settings['db_folder'],
        )
        (range_update_directives,
         issues) = get_range_directives_gatecharacterization(
            fit_range_update_directives,
            self.current_ranges,
            [self.gate.safety_range()],
        )

        return range_update_directives, issues

    def clean_up(self) -> None:
        """"""
        self.gate.dc_voltage(self.gate.safety_range()[1])

    def finish_early(self,
                     current_output_dict: Dict[str, float],
                     ) -> bool:
        """Check strength of measured signal over the last 30mv and
        see if current is constantly low/high. Measurement will be stopped
        of this is the case
        """
        readout_method_to_use = 'dc_current'
        param = self.readout_methods[readout_method_to_use]
        current_signal = current_output_dict[param.full_name]
        finish = False
        voltage_precision = self.setpoint_settings['voltage_precision']

        self._recent_signals.append(current_signal)
        if len(self._recent_signals) > int(0.3 / (voltage_precision)):
            self._recent_signals = self._recent_signals[1:].copy()

            norm_consts = self.data_settings['normalization_constants']
            n_cts = norm_consts[readout_method_to_use]
            norm_signal = (current_signal - n_cts[0]) / (n_cts[1] - n_cts[0])
            if norm_signal < self.noise_level:
                finish = True

        return finish

    def additional_post_measurement_actions(self) -> None:
        """"""
        self._recent_signals = []
