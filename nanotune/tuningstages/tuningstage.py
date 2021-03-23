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
        print(f'tuningstage readout methods: {readout_methods}')
        self.measurement_options = measurement_options
        if fit_options is None:
            fit_options = {}
        self.fit_options = fit_options

        self.stage = stage

        self.update_settings = update_settings

        self.current_ranges: List[Tuple[float, float]] = []
        self.current_setpoints: List[List[float]] = []
        self.result_ids: List[int] = []
        self.failed_ids: List[int] = []
        self.current_id: int = -1
        self.max_count = 10
        self.dv: List[float] = []

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
    def update_measurement_settings(
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

    def do_at_outer_setpoint(
        self,
        current_setpoint: float,
    ) -> None:
        """"""
        pass

    def finish_early(
        self,
        current_output_dict: Dict[str, float],
        readout_methods_to_use: str = 'dc_current',
    ) -> bool:
        """"""
        return False

    def compute_setpoints(self) -> None:
        """
        Swaps start and stop point if necessary but keeps self.current_ranges
        as they are
        """
        print(self.readout_methods)
        gates_to_sweep = self.setpoint_settings['gates_to_sweep']
        for gg, c_range in enumerate(self.current_ranges):
            diff1 = abs(c_range[1] - gates_to_sweep[gg].dc_voltage())
            diff2 = abs(c_range[0] - gates_to_sweep[gg].dc_voltage())

            if diff1 < diff2:
                self.current_ranges[gg] = (c_range[1], c_range[0])

        self.current_setpoints = []
        self.n_points = []
        self.dv = []
        voltage_precision = self.setpoint_settings['voltage_precision']

        for read_method in self.readout_methods.keys():
            standard_n = nt.config["core"]["standard_shapes"][str(self._D)]
            for gg, c_range in enumerate(self.current_ranges):
                # Calculate the number of points we need to cover the entire
                # range without exceeding the specified gate.max_jump.
                delta = abs(c_range[1] - c_range[0])
                n_safe = int(floor(delta / gates_to_sweep[gg].max_jump()))
                n_precise = int(floor(delta / voltage_precision))

                n = np.max([n_safe, n_precise, standard_n[gg]])
                setpoints = np.linspace(c_range[0], c_range[1], n)
                self.n_points.append(int(n))
                self.current_setpoints.append(setpoints)
                self.dv.append(delta / (n - 1))

            if read_method == "rf":
                raise NotImplementedError

    def take_data(self) -> int:
        """"""
        self.compute_setpoints()
        qc_measurement_parameters = list(self.readout_methods.values())
        return self._take_data(qc_measurement_parameters)

    @contextmanager
    def set_up_gates_for_measurement(self) -> Generator[None, None, None]:
        """ """
        for gg, gate in enumerate(self.setpoint_settings['gates_to_sweep']):
            # Ramp gates to start values before turning off ramping:
            gate.dc_voltage(self.current_setpoints[gg][0])
            # deactivate ramp, setpoints are calculated such that
            # voltage differences do not exceed max_ramp
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

    def _take_data(self, qc_measurement_parameters: List[qc.Parameter]) -> int:
        """
        It will always sweep the same gates and measure the same parameter
        TO DO: Implement smart way of sampling measurement points
        """
        meas = QC_Measurement()
        output = []
        output_dict: Dict[str, Optional[float]] = {}
        gate_parameters = []
        n_points_true = [0, 0]
        gates_to_sweep = self.setpoint_settings['gates_to_sweep']

        nt.set_database(self.data_settings['db_name'],
                        db_folder=self.data_settings['db_folder'])

        nt_meta = self._prepare_nt_metadata()

        with self.set_up_gates_for_measurement():
            for gate in gates_to_sweep:
                meas.register_parameter(gate.dc_voltage)
                gate_parameters.append(gate.dc_voltage)

            for m_param in qc_measurement_parameters:
                _flush_buffers(m_param)
                meas.register_parameter(m_param, setpoints=gate_parameters)
                output.append([m_param, None])
                output_dict[m_param.full_name] = None

            start_time = time.time()
            done = False

            with meas.run() as datasaver:
                # Save some important metadata before we start measuring
                datasaver.dataset.add_metadata(nt.meta_tag, json.dumps(nt_meta))

                for set_point0 in self.current_setpoints[0]:
                    gates_to_sweep[0].dc_voltage(set_point0)
                    self.do_at_outer_setpoint(set_point0)
                    n_points_true[0] += 1

                    if len(gates_to_sweep) == 2:
                        gates_to_sweep[1].use_ramp(True)
                        start_voltage = self.current_setpoints[1][0]

                        gates_to_sweep[1].dc_voltage(start_voltage)
                        gates_to_sweep[1].use_ramp(False)

                        for set_point1 in self.current_setpoints[1]:
                            gates_to_sweep[1].dc_voltage(set_point1)
                            n_points_true[1] += 1
                            m_params = qc_measurement_parameters
                            for p, parameter in enumerate(m_params):
                                value = parameter.get()
                                output[p][1] = value
                                output_dict[parameter.full_name] = value

                            paramx = gates_to_sweep[0].dc_voltage.full_name
                            paramy = gates_to_sweep[1].dc_voltage.full_name
                            datasaver.add_result(
                                (paramx, set_point0),
                                (paramy, set_point1),
                                *output, # type: ignore
                            )
                            done = self.finish_early(output_dict)  # type: ignore
                            if done:
                                break
                    else:
                        m_params = qc_measurement_parameters
                        for p, parameter in enumerate(m_params):
                            value = parameter.get()
                            output[p][1] = value
                            output_dict[parameter.full_name] = value

                        paramx = gates_to_sweep[0].dc_voltage.full_name
                        datasaver.add_result(
                            (paramx, set_point0), *output # type: ignore
                        )
                        done = self.finish_early(output_dict)  # type: ignore
                    if done:
                        break

                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(elapsed_time, 60)
                msg = "Elapsed time to take data: {:.0f} min, {:.2f} sec."
                logger.info(msg.format(minutes, seconds))

                # Add last bits of info to metadata
                nt_meta["n_points"] = n_points_true
                nt_meta["elapsed_time"] = round(float(elapsed_time), 2)

                datasaver.dataset.add_metadata(nt.meta_tag, json.dumps(nt_meta))

        return datasaver.run_id

    def _run_stage(
        self,
        plot_measurements: bool = True,
    ) -> Tuple[bool, List[str]]:
        """"""
        done = False
        count = 0
        issues = None
        termination_reasons: List[str] = []

        while not done:
            count += 1
            logger.info("Iteration no " + str(count))
            self.current_id = self.take_data()
            self.extract_features()

            self.additional_post_measurement_actions()

            self.current_quality = self.check_quality()
            self.save_predicted_category()
            if plot_measurements:
                self.current_fit.plot_fit()
                plt.pause(0.05)

            if self.current_quality or not self.update_settings:
                self.result_ids.append(self.current_id)
                if self.current_quality:
                    logger.info("Good result found.")
                else:
                    qual = "good" if self.current_quality else "poor"
                    logger.info(" Proceeding with {} quality.".format(qual))
                done = True
                success = bool(self.current_quality)
                termination_reasons = []
            else:
                self.failed_ids.append(self.current_id)
                actions, issues = self.get_next_actions()
                logger.info(f'{self.stage}: Next actions are {actions}.')

                if not actions:
                    logger.info("Hopeless case.")
                    done = True
                    success = False
                    termination_reasons = issues
                else:
                    self.update_measurement_settings(actions)
                    done = False
                    success = False
                    termination_reasons = []

            if count >= self.max_count:
                self.result_ids.append(self.current_id)
                logger.info(f"{self.stage}: max count achieved.")
                done = True
                success = False
                if issues is not None:
                    termination_reasons = issues
                termination_reasons.append("max count achieved")

        self.clean_up()
        return success, termination_reasons

    def run_stage(
        self,
        plot_measurements: bool = True,
    ) -> TuningResult:
        """
        result returns the valid range and transition voltage for the gate
        swept.
        result = {
        gate_id: [[low_v, high_v], trans_v]
        }
        """
        p_m = plot_measurements
        success, termination_reasons = self._run_stage(plot_measurements=p_m)

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


def _flush_buffers(*params: Any):
    """
    If possible, flush the VISA buffer of the instrument of the
    provided parameters. The params can be instruments as well.
    This ensures there is no stale data read off...
    Supposed to be called inside linearNd like so:
    _flush_buffers(inst_set, *inst_meas)
    """

    for param in params:
        if hasattr(param, "_instrument"):
            inst = param._instrument
        elif isinstance(param, VisaInstrument):
            inst = param
        else:
            inst = None

        if inst is not None and hasattr(inst, "visa_handle"):
            status_code = inst.visa_handle.clear()
            if status_code is not None:
                logger.warning(
                    "Cleared visa buffer on "
                    "{} with status code {}".format(inst.name, status_code)
                )
