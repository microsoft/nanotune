from typing import Optional, Tuple, List, Union, Dict, Callable, Any, Sequence
import logging
import copy
import qcodes as qc

import nanotune as nt
from nanotune.device.gate import Gate
from nanotune.fit.pinchofffit import PinchoffFit
from nanotune.tuningstages.tuningstage import TuningStage
from nanotune.classification.classifier import Classifier

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
        measurement_options: Optional[Dict[str, Dict[str, Any]]] = None,
        fit_options: Optional[Dict[str, Any]] = None,
        update_settings: bool = True,
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
            measurement_options=measurement_options,
            update_settings=update_settings,
            fit_options=fit_options,
        )

        self.clf = classifier
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

    def check_quality(self) -> bool:
        """"""
        found_dots = self.clf.predict(
            self.current_id,
            self.data_settings['db_name'],
            self.data_settings['db_folder'],
            )
        return any(found_dots)

    def update_current_ranges(
        self,
        range_update_directives: List[str],
    ) -> None:
        """"""
        for directives in range_update_directives:
            if directives not in ["x more negative", "x more positive"]:
                logger.error((f'{self.stage}: Unknown range update directives.'
                    'Cannot update measurement setting'))

        if "x more negative" in range_update_directives:
            self._update_range(0, 0)
        if "x more positive" in range_update_directives:
            self._update_range(0, 1)


    def _update_range(self, gate_id, range_id):
        v_change = abs(
            self.current_ranges[gate_id][range_id]
            - self.gate.safety_range()[range_id]
        )
        sign = (-1) ** (range_id + 1)
        self.current_ranges[gate_id][range_id] += sign * v_change

    def get_range_update_directives(self) -> Tuple[List[str], List[str]]:
        """
        Define range_update_directives if quality of current fit is sub-optimal.
        First: try to sweep the current gate more negative.
        If we are speeing to its min_v, we set the auxiliary_gate to min_v
        No intermediate tries, just being efficient.
        """
        fit_range_update_directives = self.current_fit.range_update_directives
        safety_range = self.gate.safety_range()

        neg_range_avail = abs(self.current_ranges[0][0] - safety_range[0])
        pos_range_avail = abs(self.current_ranges[0][1] - safety_range[1])

        range_update_directives = []
        issues = []

        if "x more negative" in fit_range_update_directives:
            if neg_range_avail >= 0.1:
                range_update_directives.append("x more negative")
            else:
                issues.append("negative safety voltage reached")

        if "x more positive" in fit_range_update_directives:
            if pos_range_avail >= 0.1:
                range_update_directives.append("x more positive")
            else:
                issues.append("positive safety voltage reached")

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
