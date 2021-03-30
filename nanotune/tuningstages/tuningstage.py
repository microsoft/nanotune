import logging
import json
import time
import datetime
import copy
import numpy as np
from math import floor
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, List, Dict, Any, Sequence, Union, Generator
from contextlib import contextmanager

import matplotlib.pyplot as plt
import qcodes as qc
from qcodes.dataset.measurements import Measurement as QC_Measurement
from qcodes.dataset.experiment_container import load_by_id
from qcodes.instrument.visa import VisaInstrument
from nanotune.device_tuner.tuningresult import TuningResult
import nanotune as nt
from nanotune.device.gate import Gate
from .take_data import take_data, ramp_to_setpoint, compute_linear_setpoints
logger = logging.getLogger(__name__)
SETPOINT_METHODS = nt.config["core"]["setpoint_methods"]
data_dimensions = {
    'gatecharacterization1d': 1,
    'chargediagram': 2,
    'coulomboscillations': 1,
}

# readout_methods = {'dc_current': qc.Parameter,
#                     'dc_sensor': qc.Parameter}
# }
# setpoint_settings = {
#     'voltage_precision':
#     'gates_to_sweep':
# }
# measurement_options = {
#       'dc_current': {'delay': 0.1,
#                       'inter_delay': 0.1,
#   }
# }
# data_settings = {
#     'db_name': '',
#     'normalization_constants': {},
#     'db_folder': '',
# }
# fit_options = {},


class TuningStage(metaclass=ABCMeta):
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
        self.current_id: int = -1
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
    def get_next_actions(self) -> Tuple[List[str], List[str]]:
        """
        The fit is telling us what we need to to
        Return "false" as second argument if we should abandon the case
        """
        pass

    @abstractmethod
    def update_current_ranges(
        self,
        actions: List[str],
    ) -> None:
        """"""

    def extract_features(self):
        """"""
        self.current_fit = self.fit_class(
            self.current_id, self.data_settings['db_name'],
            **self.fit_options,
            db_folder=self.data_settings['db_folder'],
        )
        self.current_fit.find_fit()
        self.current_fit.save_features()

    @abstractmethod
    def check_quality(self) -> bool:
        """"""

    def save_predicted_category(self):
        """"""
        ds = load_by_id(self.current_id)
        nt_meta = json.loads(ds.get_metadata(nt.meta_tag))

        nt_meta["predicted_category"] = int(self.current_quality)
        ds.add_metadata(nt.meta_tag, json.dumps(nt_meta))

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
        for gg, c_range in enumerate(self.current_ranges):
            diff1 = abs(c_range[1] - gates_to_sweep[gg].dc_voltage())
            diff2 = abs(c_range[0] - gates_to_sweep[gg].dc_voltage())

            if diff1 < diff2:
                self.current_ranges[gg] = (c_range[1], c_range[0])

        setpoints = compute_linear_setpoints(
            self.current_ranges,
            self.setpoint_settings['voltage_precision'],
            max_jumps=[gate.max_jump() for gate in gates_to_sweep],
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

    def _prepare_nt_metadata(self) -> Dict[str, Any]:
        gates_to_sweep = self.setpoint_settings['gates_to_sweep']
        nt_meta = dict.fromkeys(nt.config["core"]["meta_fields"])

        norm = self.data_settings['normalization_constants']
        nt_meta["normalization_constants"] = norm
        nt_meta["git_hash"] = nt.git_hash
        nt_meta["device_name"] = gates_to_sweep[0].parent.name
        m_jumps = [g.max_jump() for g in gates_to_sweep]
        nt_meta["max_jumps"] = m_jumps
        ramp_rates = [g.ramp_rate() for g in gates_to_sweep]
        nt_meta["ramp_rate"] = ramp_rates
        r_meth = self.readout_methods
        read_dict = {k:param.full_name for (k, param) in r_meth.items()}
        nt_meta["readout_methods"] = read_dict
        nt_meta["features"] = {}
        return nt_meta

    def measure(self) -> int:
        """
        """
        if not self.current_setpoints:
            self.current_setpoints = self.compute_setpoints()

        nt.set_database(self.data_settings['db_name'],
                        db_folder=self.data_settings['db_folder'])

        qc_measurement_parameters = list(self.readout_methods.values())

        parameters_to_sweep = []
        for gate in self.setpoint_settings['gates_to_sweep']:
            parameters_to_sweep.append(gate.dc_voltage)

        with self.set_up_gates_for_measurement():
            start_time = time.time()
            run_id, n_measured = take_data(
                parameters_to_sweep,
                qc_measurement_parameters,
                self.current_setpoints,
                finish_early_check=self.finish_early,
                do_at_inner_setpoint=ramp_to_setpoint,
                metadata_addon=(nt.meta_tag, self._prepare_nt_metadata())
            )

            ds = load_by_id(run_id)
            nt_meta = json.loads(ds.get_metadata(nt.meta_tag))
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            msg = "Elapsed time to take data: {:.0f} min, {:.2f} sec."
            logger.info(msg.format(minutes, seconds))

            # Add last bits of info to metadata
            nt_meta["n_points"] = n_measured
            nt_meta["elapsed_time"] = round(float(elapsed_time), 2)

            ds.add_metadata(nt.meta_tag, json.dumps(nt_meta))

        return run_id

    def _run_stage(
        self,
        plot_measurements: bool = True,
    ) -> Tuple[bool, List[str]]:
        """"""
        done = False
        count = 0
        termination_reasons: List[str] = []

        while not done:
            count += 1
            logger.info("Iteration no " + str(count))

            self.current_setpoints = self.compute_setpoints()
            self.current_id = self.measure()

            self.result_ids.append(self.current_id)
            self.extract_features()
            self.additional_post_measurement_actions()

            self.current_quality = self.check_quality()
            self.save_predicted_category()

            if self.current_quality:
                logger.info("Good result found.")
            else:
                logger.info("Poor quality.")

            if plot_measurements:
                self.current_fit.plot_fit()
                plt.pause(0.05)

            success = bool(self.current_quality)
            if success:
                done = True
                termination_reasons = []
            elif self.update_settings:
                actions, termination_reasons = self.get_next_actions()
                logger.info(
                    self.stage + 'next actions: ' + ', '.join(actions)
                )
                if not actions:
                    done = True
                    success = False
                else:
                    self.update_current_ranges(actions)
                    done = False
                    success = False

            if count >= self.max_count:
                logger.info(f"{self.stage}: max count reached.")
                done = True
                success = False
                termination_reasons.append("max count reached")

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
            features=self.current_fit.features,
            timestamp=datetime.datetime.now().isoformat(),
        )

        return tuning_result
