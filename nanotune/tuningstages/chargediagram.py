import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import TypedDict

from nanotune.device_tuner.tuningresult import TuningResult
from nanotune.device.device import Readout
from nanotune.fit.dotfit import DotFit
from nanotune.tuningstages.tuningstage import TuningStage
from nanotune.tuningstages.settings import (DataSettings, SetpointSettings,
    Classifiers)

from .base_tasks import (  # please update docstrings if import path changes
    conclude_iteration_with_range_update,
    get_extracted_features, get_fit_range_update_directives)
from .chargediagram_tasks import (classify_dot_segments,
                                  conclude_dot_classification,
                                  determine_dot_regime,
                                  get_dot_segment_regimes,
                                  get_new_chargediagram_ranges,
                                  get_range_directives_chargediagram,
                                  segment_dot_data, translate_dot_regime,
                                  verify_dot_classification)

RangeChangeSettingsDict = TypedDict(
    "RangeChangeSettingsDict",
    {
        "relative_range_change": float,
        "min_change": float,
        "max_change": float,
    },
)
default_range_change_settings: RangeChangeSettingsDict = {
    "relative_range_change": 0.3,
    "min_change": 0.05,
    "max_change": 0.5,
}

logger = logging.getLogger(__name__)


class ChargeDiagram(TuningStage):
    """Tuning stage measuring charge stability diagrams.

    Attributes:
        stage: String identifier indicating which stage it implements, e.g.
            gatecharacterization.
        data_settings: Dictionary with information about data, e.g. where it
            should be saved and how it should be normalized.
            Required fields are 'db_name', 'db_folder' and
            'normalization_constants', 'segment_size'.
        setpoint_settings: Dictionary with information about how to compute
            setpoints. Required keys are 'parameters_to_sweep',
            'safety_voltages', 'current_valid_ranges' and 'voltage_precision'.
        readout:
        current_valid_ranges: List of voltages ranges (tuples of floats) to
            measure.
        safety_voltage_ranges: List of satefy voltages ranges, i.e. safety limits within
            the device stays alive.
        target_regime: String indicating the desired final charge state/dot
            regime.
        range_change_settings: Dictionary with keys 'relative_range_change',
            'min_change', 'max_change'. Used to determine new voltage ranges to
            sweep.
        classifiers: Pre-trained nt.Classifiers predicting charge states and
            their quality.
        fit_class: Returns the class used to perform data fitting, i.e.
            nanotune.fit.dotfit.Dotfit.

    """

    def __init__(
        self,
        data_settings: DataSettings,
        setpoint_settings: SetpointSettings,
        readout: Readout,
        classifiers: Classifiers,
        target_regime: str = "doubledot",
        range_change_settings: Optional[RangeChangeSettingsDict] = None,
    ) -> None:
        """Initializes the base class of a tuning stage. Voltages to sweep and
        safety voltages are determined from the list of parameters in
        setpoint_settings.

        Args:
            data_settings: Dictionary with information about data, e.g. where it
                should be saved and how it should be normalized.
                Required fields are 'db_name', 'db_folder' and
                'normalization_constants', 'segment_size'.
            setpoint_settings: Dictionary with information required to compute
                setpoints. Necessary keys are 'current_valid_ranges',
                'safety_voltage_ranges', 'parameters_to_sweep' and 'voltage_precision'.
            readout:
            classifiers: Pre-trained classifiers predicting single and dot
                quality, and the dotregime. String keys indicating the type of
                classifier map onto the classifiers itself. Required keys are
                'singledot', 'doubledot', 'dotregime'.
            target_regime: String indicating the desired final charge state/dot
            regime.
        range_change_settings: Dictionary with keys 'relative_range_change',
            'min_change', 'max_change'. Used to determine new voltage ranges to
            sweep.

        """
        if data_settings.normalization_constants is None:
            raise ValueError("No normalisation constant.")

        TuningStage.__init__(
            self,
            "chargediagram",
            data_settings,
            setpoint_settings,
            readout,
        )
        if range_change_settings is None:
            range_change_settings = {}  # type: ignore
        default_range_change_settings.update(range_change_settings)  # type: ignore

        self.range_change_settings = default_range_change_settings
        self.target_regime = target_regime
        self.classifiers = classifiers

    @property
    def fit_class(self):
        """Returns nanotune's Dotfit"""
        return DotFit

    def conclude_iteration(
        self,
        tuning_result: TuningResult,
        current_valid_ranges: List[Tuple[float, float]],
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

        (
            done,
            new_voltage_ranges,
            termination_reasons,
        ) = conclude_iteration_with_range_update(
            tuning_result,
            current_valid_ranges,
            safety_voltage_ranges,
            self.get_range_update_directives,
            partial(
                get_new_chargediagram_ranges,
                range_change_settings=self.range_change_settings,
            ),
            current_iteration,
            max_n_iterations,
        )

        return done, new_voltage_ranges, termination_reasons

    def verify_machine_learning_result(
        self,
        ml_result: Dict[str, Any],
    ) -> bool:
        """Verifies if the desired regime and quality have been found.

        Args:
            ml_result: Dictionary returned by ``self.machine_learning_task``.

        Returns:
            bool: Whether the desired outcome has been found.
        """

        good_found = False
        if ml_result["regime"] == self.target_regime and ml_result["quality"]:
            good_found = True

        return good_found

    def machine_learning_task(
        self,
        run_id: int,
    ) -> Dict[str, Any]:
        """Divides original measurement into segments, which are classified
        seperately. The overall outcome, i.e. regime and quality, are
        determined using ``conclude_dot_classification`` defined in
        .chargediagram_tasks. If the desired regime is not found, the overall
        regime is the most frequent one in the dot segments.

        Args:
            run_id: QCoDeS data run ID.

        Returns:
            dict: The classification outcome, both segment wise under the key
                'dot_segments' as well as the overall outcome under 'regime'
                and 'quality'.
        """

        dot_segments = segment_dot_data(
            run_id,
            self.data_settings.db_name,
            self.data_settings.db_folder,
            self.data_settings.segment_db_name,
            self.data_settings.segment_db_folder,
            self.data_settings.segment_size,
        )

        classification_outcome = classify_dot_segments(
            self.classifiers,
            [run_id for run_id in dot_segments.keys()],
            self.data_settings.segment_db_name,
            self.data_settings.segment_db_folder,
        )
        segment_regimes = get_dot_segment_regimes(
            classification_outcome,
            determine_dot_regime,
        )

        for r_id in dot_segments.keys():
            dot_segments[r_id]["predicted_regime"] = segment_regimes[r_id]

        ml_result: Dict[str, Any] = {}
        ml_result["dot_segments"] = dot_segments

        ml_result["regime"], ml_result["quality"] = conclude_dot_classification(
            self.target_regime,
            dot_segments,
            verify_dot_classification,
            translate_dot_regime,
        )
        ml_result["features"] = get_extracted_features(
            self.fit_class,
            run_id,
            self.data_settings.db_name,
            db_folder=self.data_settings.db_folder,
        )

        return ml_result

    def get_range_update_directives(
        self,
        run_id: int,
        current_valid_ranges: List[Tuple[float, float]],
        safety_voltage_ranges: List[Tuple[float, float]],
    ) -> Tuple[List[str], List[str]]:
        """Determines directives indicating if the current voltage ranges need
        to be extended or shifted. It first gets these directives from the data
        fit using ``get_fit_range_update_directives`` defined in .base_tasks.py
        and then checks if they can be put into action using
        ``get_range_directives_chargediagram`` defined in
        chargediagram_tasks.py. The check looks at whether safety ranges
        have been reached already, or whether a voltage range extension is
        possible.

        Args:
            run_id: QCoDeS data run ID.
            current_valid_ranges: Last voltage range swept.
            safety_voltage_ranges: Safety range of gate swept.

        Returns:
            list: List with range update directives.
            list: List with issues encountered.
        """

        fit_range_update_directives = get_fit_range_update_directives(
            self.fit_class,
            run_id,
            self.data_settings.db_name,
            db_folder=self.data_settings.db_folder,
        )
        (range_update_directives, issues) = get_range_directives_chargediagram(
            fit_range_update_directives,
            current_valid_ranges,
            safety_voltage_ranges,
        )

        return range_update_directives, issues
