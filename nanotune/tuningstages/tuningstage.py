import logging
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Dict, List, Tuple, Sequence
import qcodes as qc
from qcodes.dataset.experiment_container import (load_last_experiment,
                                                 new_experiment,
                                                 load_experiment)
import nanotune as nt
from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.device.device import Readout

from .base_tasks import (  # please update docstrings if import path changes
    compute_linear_setpoints, get_current_voltages, iterate_stage, plot_fit,
    prepare_metadata, print_tuningstage_status, run_stage,
    save_extracted_features, save_machine_learning_result, set_voltages,
    take_data_add_metadata)
from .take_data import ramp_to_setpoint
from nanotune.tuningstages.settings import DataSettings, SetpointSettings

logger = logging.getLogger(__name__)


class TuningStage(metaclass=ABCMeta):
    """Base class implementing the common sequence of a tuning stage.

    Parameters:
        stage: String indicating which stage it implements, e.g.
            gatecharacterization.
        data_settings: Dictionary with information about data, e.g. where it
            should be saved and how it should be normalized.
            Required fields are 'db_name', 'db_folder' and
            'normalization_constants'.
        setpoint_settings: Dictionary with information required to compute
            setpoints. Necessary keys are 'current_valid_ranges',
            'safety_voltage_ranges', 'parameters_to_sweep' and 'voltage_precision'.
        readout: Dictionary mapping string identifiers such as
            'transport' to QCoDeS parameters measuring/returning the desired
            quantity (e.g. current throught the device).
        current_valid_ranges: List of voltages ranges (tuples of floats) to
            measure.
        safety_voltage_ranges: List of satefy voltages ranges, i.e. safety
            limits within which gates don't blow up.
        fit_class: Abstract property, to be specified in child classes. It is
            the class that should perform the data fitting, e.g. PinchoffFit.
    """

    def __init__(
        self,
        stage: str,
        data_settings: DataSettings,
        setpoint_settings: SetpointSettings,
        readout: Readout,
    ) -> None:
        """Initializes the base class of a tuning stage. Voltages to sweep and
        safety voltages are determined from the list of parameters in
        setpoint_settings.

        Args:
            stage: String identifier indicating which stage it implements, e.g.
                gatecharacterization.
            data_settings: Dictionary with information about data, e.g. where
                it should be saved and how it should be normalized.
                Required fields are 'db_name', 'db_folder' and
                'normalization_constants'.
            setpoint_settings: Dictionary with information required to compute
                setpoints. Necessary keys are 'current_valid_ranges',
                'safety_voltage_ranges', 'parameters_to_sweep' and
                'voltage_precision'.
            readout: Dataclass of DelegateParameter used for readout
        """

        self.stage = stage
        self.data_settings = data_settings
        self.setpoint_settings = setpoint_settings
        self.readout = readout

        ranges = self.setpoint_settings.ranges_to_sweep
        self.current_valid_ranges = ranges

    @property
    @abstractmethod
    def fit_class(self):
        """To be specified in child classes. It is the data fitting
        class should be used to perform a fit.
        """

    @abstractmethod
    def conclude_iteration(
        self,
        tuning_result: TuningResult,
        current_valid_ranges: Sequence[Sequence[float]],
        safety_voltage_ranges: Sequence[Sequence[float]],
        current_iteration: int,
        max_n_iterations: int,
    ) -> Tuple[bool, Sequence[Sequence[float]], List[str]]:
        """Method checking if one iteration of a run_stage measurement cycle has
        been successful.

        An iteration of such a measurement cycle takes data,
        performs a machine learning task, verifies and saves the machine
        learning result. If a repetition of this cycle is supported, then
        `conclude_iteration` determines whether another iteration should take
        place and which voltage ranges need to be measured.
        Each child class needs to implement the body of this method, tailoring
        it to the respective tuning stage.

        Args:
            tuning_result: Result of the last run_stage measurement cycle.
            current_valid_ranges: Voltage ranges last swept.
            safety_voltage_ranges: Safety voltage ranges, i.e. largest possible
                range that could be swept.
            current_iteration: Number of current iteration.
            max_n_iterations: Maximum number of iterations to perform before
                abandoning.

        Returns:
            bool: Whether this is the last iteration and the stage is done/to
                be stopped.
            list: New voltage ranges to sweep if the stage is not done.
            list: List of strings indicating failure modes.
        """

    @abstractmethod
    def verify_machine_learning_result(
        self,
        ml_result: Dict[str, int],
    ) -> bool:
        """Verifies if the desired measurement quality or regime has been found.
        Needs to be implemented by child classed to account for the different
        regimes or measurements they are dealing with.

        Args:
            ml_result: Result returned by ``machine_learning_task``.

        Returns:
            bool: Whether the desired outcome has been found.
        """

    @abstractmethod
    def machine_learning_task(
        self,
        run_id: int,
    ) -> Dict[str, Any]:
        """The machine learning task to perform after a measurement.

        Args:
            run_id: QCoDeS data run ID.
        """

    def save_ml_result(
        self,
        run_id: int,
        ml_result: Dict[str, int],
    ) -> None:
        """Saves the result returned by ```machine_learning_task```: the
        extracted features are stored into metadata of the respective dataset.

        Args:
            run_id: QCoDeS data run ID.
            ml_result: Result returned by ``machine_learning_task``.
        """

        save_extracted_features(
            self.fit_class,
            run_id,
            self.data_settings.db_name,
            db_folder=self.data_settings.db_folder,
        )
        save_machine_learning_result(run_id, ml_result)

    def finish_early(
        self,
        current_output_dict: Dict[str, float],
    ) -> bool:
        """Checks if the current data taking can be stopped. E.g. if the device
        is pinched off entirely.

        Args:
            current_output_dict: Dictionary mapping a string indicating the
                readout method to the respective value last measured.

        Returns:
            bool: Whether the current data taking procedure can be stopped.
        """

        return False

    def compute_setpoints(
        self,
        current_valid_ranges: Sequence[Sequence[float]],
    ) -> Sequence[Sequence[float]]:
        """Computes setpoints for the next measurement. Unless this method is
        overwritten in a child class, linearly spaced setpoints are computed.

        Args:
            current_valid_ranges: Voltages ranges to sweep.

        Returns:
            list: List of lists with setpoints.
        """

        setpoints = compute_linear_setpoints(
            current_valid_ranges,
            self.setpoint_settings.voltage_precision,
        )
        return setpoints

    def show_result(
        self,
        plot_result: bool,
        current_id: int,
        tuning_result: TuningResult,
    ) -> None:
        """Displays tuning result and optionally plots the fitting result.

        Args:
            plot_result: Bool indicating whether the data fit should be
                plotted.
            current_id: QCoDeS data run ID.
            tuning_result: Result of a tuning stage run.
        """

        if plot_result:
            plot_fit(
                self.fit_class,
                current_id,
                self.data_settings.db_name,
                db_folder=self.data_settings.db_folder,
            )
        print_tuningstage_status(tuning_result)

    def prepare_nt_metadata(self) -> Dict[str, Any]:
        """Sets up a metadata dictionary with fields known prior to a
        measurement set. Wraps ```prepare_metadata``` in .base_tasks.py.

        Returns:
            dict: Metadata dict with fields known prior to a measurement filled
                in.
        """
        example_param = self.setpoint_settings.parameters_to_sweep[0]
        device_name = example_param.name_parts[0]
        nt_meta = prepare_metadata(
            device_name,
            self.data_settings.normalization_constants,
            self.readout,
        )
        return nt_meta

    def measure(
        self,
        parameters_to_sweep: List[qc.Parameter],
        parameters_to_measure: List[qc.Parameter],
        setpoints: List[List[float]],
    ) -> int:
        """Takes 1D or 2D data and saves relevant metadata into the dataset.
        Wraps ```take_data_add_metadata``` in .base_tasks.py.

        Args:
            setpoints: Setpoints to measure.

        Returns:
            int: QCoDeS data run ID.
        """

        run_id = take_data_add_metadata(
            parameters_to_sweep,
            parameters_to_measure,
            setpoints,
            finish_early_check=self.finish_early,
            do_at_inner_setpoint=ramp_to_setpoint,
            pre_measurement_metadata=self.prepare_nt_metadata(),
        )

        return run_id

    def run_stage(
        self,
        iterate: bool = True,
        max_iterations: int = 10,
        plot_result: bool = True,
    ) -> TuningResult:
        """Performs iterations of a basic measurement cycle of a tuning stage.

        It wraps ```iterate_stage``` in .base_tasks.py. One measurement cycle
        does the following subtasks:
        - computes setpoints
        - perform the actual measurement, i.e. take data
        - perform a machine learning task, e.g. classification
        - validate the machine learning result, e.g. check if a good regime was found
        - collect all information in a TuningResult instance.
        At each iteration, ```conclude_iteration``` check whether another
        measurement cycle will be performed.
        At the very end, ```clean_up``` does the desired post-measurement task.

        Args:
            iterate:
            max_iterations:
            plot_result:

        Returns:
            TuningResult: Tuning results of the last iteration, with the
                dataids field containing QCoDeS run IDs of all datasets
                measured.
        """

        nt.set_database(
            self.data_settings.db_name,
            db_folder=self.data_settings.db_folder
        )
        if self.data_settings.experiment_id is None:
            try:
                qcodes_experiment = load_last_experiment()
            except ValueError:
                logger.warning(
                    "No qcodes experiment found. Starting a new "
                    'one called "automated_tuning", with an unknown sample.'
                )
                qcodes_experiment = new_experiment(
                    "automated_tuning", sample_name="unknown"
                )
            self.data_settings.experiment_id = qcodes_experiment.exp_id
        else:
            load_experiment(self.data_settings.experiment_id)

        initial_voltages = get_current_voltages(
            self.setpoint_settings.parameters_to_sweep
        )

        run_stage_tasks = [
            self.compute_setpoints,
            self.measure,
            self.machine_learning_task,
            self.save_ml_result,
            self.verify_machine_learning_result,
        ]
        if not iterate:
            max_iterations = 1

        tuning_result = iterate_stage(
            self.stage,
            self.setpoint_settings.parameters_to_sweep,
            self.readout.get_parameters(),
            self.current_valid_ranges,
            self.setpoint_settings.safety_voltage_ranges,
            run_stage,
            run_stage_tasks,  # type: ignore
            self.conclude_iteration,
            partial(self.show_result, plot_result),
            max_iterations,
        )
        set_voltages(
            self.setpoint_settings.parameters_to_sweep,
            initial_voltages,
        )

        tuning_result.db_name = self.data_settings.db_name
        tuning_result.db_folder = self.data_settings.db_folder

        return tuning_result
