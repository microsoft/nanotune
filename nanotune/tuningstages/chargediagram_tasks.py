# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import logging
import copy
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Any,
    Sequence,
    Union,
    Callable,
)
from typing_extensions import TypedDict
import nanotune as nt
from nanotune.fit.dotfit import DotFit
from nanotune.classification.classifier import Classifier

DotClassifierOutcome = TypedDict(
    "DotClassifierOutcome",
    {
        "singledot": int,
        "doubledot": int,
        "dotregime": int,
    },
)
DotClassifierDict = TypedDict(
    "DotClassifierDict",
    {
        "singledot": Classifier,
        "doubledot": Classifier,
        "dotregime": Classifier,
    },
)
logger = logging.getLogger(__name__)
DOT_LABEL_MAPPING = dict(nt.config["core"]["dot_mapping"])


def segment_dot_data(
    run_id: int,
    db_name: str,
    db_folder: Optional[str] = None,
    segment_db_name: Optional[str] = None,
    segment_db_folder: Optional[str] = None,
    segment_size: float = 0.05,
) -> Dict[int, Any]:
    """Divides a 2D measurement into segments of segment_size x segment_size
    in Volt.

    Args:
        run_id: QCoDeS data run ID.
        db_name: Name of database containg dataset.
        db_folder: Path to folder containing db_name.
        segment_db_name: Name of database where data segments should be saved.
        segment_db_folder: Path to folder containing segment_db_name.
        segment_size: Voltage interval the segments should span in each
            dimension.

    Returns:
        dict: Dictionary mapping run IDs of segmented data to the voltages
            ranges they span:
            dot_segments = {
                <run_id>: {
                    'voltage_ranges': [(), ()],
                }
            }.

    """

    if segment_db_name is None:
        segment_db_name = "segmented_" + db_name

    fit = DotFit(
        run_id,
        db_name,
        db_folder=db_folder,
        segment_size=segment_size,
    )
    dot_segments = fit.save_segmented_data_return_info(
        segment_db_name,
        segment_db_folder,
    )

    return dot_segments


def classify_dot_segments(
    classifiers: DotClassifierDict,
    run_ids: List[int],
    db_name: str,
    db_folder: Optional[str] = None,
) -> Dict[int, Dict[str, Union[bool, int]]]:
    """Classifies several datasets holding charge diagrams, e.g. previously
    segmented dot data.

    Args:
        classifiers: Pre-trained classifiers predicting single and double dot
            quality and dotregime.
        run_ids: QCoDeS data run IDs.
        db_name: Name of database where datasets are saved.
        db_folder: Path to folder containing segment_db_name.

    Returns:
        dict: Mapping run IDs onto a dict holding the respective classification
            result of all classifiers passed. For example:
            clf_result = {
                <run_id>: {
                    'singledot': True,
                    'doubledot': False,
                    'dotregime': False,
                }
            }.
    """

    if db_folder is None:
        db_folder = nt.config["db_folder"]

    with nt.switch_database(db_name, db_folder):
        clf_result: Dict[int, Dict[str, Union[bool, int]]] = {}
        for data_id in run_ids:
            clf_result[data_id] = {}
            for clf_type, classifier in classifiers.items():
                clf_result[data_id][clf_type] = any(
                    classifier.predict(  # type: ignore
                        data_id,
                        db_name,
                        db_folder=db_folder,
                    )
                )

    return clf_result


def get_dot_segment_regimes(
    dot_segments_classification_result: Dict[int, Dict[str, int]],
    determine_regime: Callable[[Dict[str, Union[bool, int]]], int],
) -> Dict[int, int]:
    """Takes the classification results of several datasets, merging the
    predictions of several classifiers into a single regime indicator. The
    decision about how predictions is unified is implented in
    ``resolve_regime``.

    Args:
        dot_segments_classification_result: Classification predictions of
            datasets. Each run_id maps onto a dictionary holding multiple
            prediction outcomes, e.g.
            dot_segments_classification_result = {
                <run_id>: {
                    'singledot': True,
                    'doubledot': False,
                    'dotregime': False,
                }
            }.
        determine_regime: Function merging several classification outcomes into
            a single regime indicator.

    Returns:
        dict: Mapping run_ids on a single regime indicator.
    """

    segment_regimes = {}
    for data_id, clf_result in dot_segments_classification_result.items():
        regime = determine_regime(clf_result)
        segment_regimes[data_id] = regime

    return segment_regimes


def determine_dot_regime(
    classification_result: Dict[str, Union[bool, int]],
) -> int:
    """Determines the dot regime based on the classification outcome of
    single and double dot quality and dot regime predictions.

    Args:
        classification_result: Single and double dot quality, and dotregime
            predictions of a single dataset. Required items are 'singledot',
            'doubledot' and 'dotregime' keys mapping onto their respective
            classifiers.

    Returns:
        int: Regime indicator, according to the regime mapping defined in
            nt.config["core"]["dot_mapping"].
    """

    clfs = ["singledot", "doubledot", "dotregime"]
    if not all(clf in classification_result.keys() for clf in clfs):
        raise KeyError("Classification outcome missing, unable to determine dot regime")

    good_single = bool(classification_result["singledot"])
    good_double = bool(classification_result["doubledot"])
    dotregime = bool(classification_result["dotregime"])

    if good_single and good_double:
        # good single and good double dot
        if dotregime:
            regime = DOT_LABEL_MAPPING["doubledot"][1]
        else:
            regime = DOT_LABEL_MAPPING["singledot"][1]
    elif not good_single and not good_double:
        if dotregime:
            regime = DOT_LABEL_MAPPING["doubledot"][0]
        else:
            regime = DOT_LABEL_MAPPING["singledot"][0]
    elif good_single and not good_double:
        regime = DOT_LABEL_MAPPING["singledot"][1]
    elif not good_single and good_double:
        regime = DOT_LABEL_MAPPING["doubledot"][1]
    else:
        raise ValueError("Unable to resolve dot regime.")

    return regime


def translate_dot_regime(
    regime: int,
) -> Tuple[str, bool]:
    """Takes an integer regime indicator and returns the corresponding charge
    state and quality as a string and boolean respectively. Uses the dot
    regime mapping defined in nt.config["core"]["dot_mapping"].

    Args:
        regime: Dot regime indicator.

    Returns:
        str: Charge state, e.g. 'singledot'.
        bool: quality.

    """

    rev_mapping = {}
    for str_regime in ["singledot", "doubledot"]:
        idx = DOT_LABEL_MAPPING[str_regime]
        rev_mapping[idx[0]] = (str_regime, False)
        rev_mapping[idx[1]] = (str_regime, True)

    return rev_mapping[regime]


def verify_dot_classification(
    target_regime: str,
    regimes: Sequence[int],
) -> bool:
    """Verifies if the target regime has been found.

    Args:
        target_regime: E.g. 'doubledot'.
        regimes: List of regimes (int indicators) found within a diagram.

    Returns:
        bool: Whether or not the target regime has been found.
    """

    good_found = False
    target = DOT_LABEL_MAPPING[target_regime][1]
    if target in regimes:
        good_found = True
    return good_found


def conclude_dot_classification(
    target_charge_state: str,
    dot_segments: Dict[int, Any],
    verify_classification_outcome: Callable[[str, Sequence[int]], bool],
    interpret_single_outcome: Callable[[int], Tuple[str, bool]],
) -> Tuple[str, bool]:
    """Determines the charge state and quality of a charge diagram based on
    classification outcomes of its segments/sub-measurements.

    Args:
        target_charge_state: Target charge state
        dot_segments: Dictionary mapping run_ids of dot segments onto their
            predicted regime:
            dot_segments= {
                    <run_id>: {
                        'predicted_regime': int,
                        }
                    }
        verify_classification_outcome: Function checking whether the target
            regime has been found.
        interpret_single_outcome: Interprets the classification result (int) to
            return a string and bool indicating the charge state and quality.

    Returns:
        str: Charge state, e.g. 'singledot'.
        bool: quality.
    """

    dot_segment_vals = dot_segments.values()
    regimes = [seg["predicted_regime"] for seg in dot_segment_vals]
    good_found = verify_classification_outcome(target_charge_state, regimes)

    if good_found:
        regime = target_charge_state
        quality = True
    else:
        majority_vote = max(regimes, key=regimes.count)
        regime, quality = interpret_single_outcome(majority_vote)

    return regime, quality


def get_range_directives_chargediagram(
    fit_range_update_directives: List[str],
    current_valid_ranges: List[Tuple[float, float]],
    safety_voltage_ranges: List[Tuple[float, float]],
) -> Tuple[List[str], List[str]]:
    """Determines voltage range directives to update ranges for a subsequent
    tuning stage iteration. It checks if the voltage range update directives
    determined previously, e.g by a fit class, can be carried out based on the
    supplied safety ranges.

    Args:
        fit_range_update_directives: Directives determined previously, such as
            during fitting.
        current_valid_ranges: Voltage range swept at previous iteration.
        safety_voltage_ranges: Safety range of gate/voltage parameter swept.

    Returns:
        list: Range update directives, e.g. 'x more negative'.
        list: Issues encountered, e.g. 'positive safety voltage reached'.
    """

    issues = []
    range_update_directives = []
    if "x more negative" in fit_range_update_directives:
        d_n = abs(current_valid_ranges[0][0] - safety_voltage_ranges[0][0])
        if d_n > 0.015:
            range_update_directives.append("x more negative")
        else:
            issues.append("x reached negative voltage limit")

    if "x more positive" in fit_range_update_directives:
        d_p = abs(current_valid_ranges[0][1] - safety_voltage_ranges[0][1])
        if d_p > 0.015:
            range_update_directives.append("x more positive")
        else:
            issues.append("x reached positive voltage limit")

    if "y more negative" in fit_range_update_directives:
        d_n = abs(current_valid_ranges[1][0] - safety_voltage_ranges[1][0])
        if d_n > 0.015:
            range_update_directives.append("y more negative")
        else:
            issues.append("y reached negative voltage limit")

    if "y more positive" in fit_range_update_directives:
        d_p = abs(current_valid_ranges[1][1] - safety_voltage_ranges[1][1])
        if d_p > 0.015:
            range_update_directives.append("y more positive")
        else:
            issues.append("y reached positive voltage limit")

    return range_update_directives, issues


def get_new_chargediagram_ranges(
    current_valid_ranges: List[Tuple[float, float]],
    safety_voltage_ranges: List[Tuple[float, float]],
    range_update_directives: List[str],
    range_change_settings: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, float]]:
    """Determines new voltage ranges for a subsequent tuning stage
    iteration. Ranges are shifted based on ``range_update_directives`` and
    ``range_change_settings``, and using ``get_new_range``.

    Args:
        current_valid_ranges: Current voltage ranges.
        safety_voltage_ranges: List of safety ranges.
        range_update_directives: List of range update directives.

    Returns:
        Tuple: New voltage range.
    """

    if range_change_settings is None:
        range_change_settings = {
            "relative_range_change": 0.3,
            "min_change": 0.05,
            "max_change": 0.5,
        }

    all_range_update_directives = [
        "x more negative",
        "x more positive",
        "y more negative",
        "y more positive",
    ]
    new_ranges = copy.deepcopy(current_valid_ranges)

    for directive in range_update_directives:
        if directive not in all_range_update_directives:
            logger.error(
                ("ChargeDiagram: Unknown action." "Cannot update measurement setting")
            )

    if "x more negative" in range_update_directives:
        new_ranges[0] = get_new_range(
            current_valid_ranges[0],
            safety_voltage_ranges[0],
            "negative",
            **range_change_settings,
        )

    if "x more positive" in range_update_directives:
        new_ranges[0] = get_new_range(
            current_valid_ranges[0],
            safety_voltage_ranges[0],
            "positive",
            **range_change_settings,
        )

    if "y more negative" in range_update_directives:
        new_ranges[1] = get_new_range(
            current_valid_ranges[1],
            safety_voltage_ranges[1],
            "negative",
            **range_change_settings,
        )

    if "y more positive" in range_update_directives:
        new_ranges[1] = get_new_range(
            current_valid_ranges[1],
            safety_voltage_ranges[1],
            "positive",
            **range_change_settings,
        )
    else:
        logger.error(
            (
                "ChargeDiagram: Unknown range update directive."
                "Cannot update measurement setting"
            )
        )

    return new_ranges


def get_new_range(
    current_valid_range: Tuple[float, float],
    safety_voltage_ranges: Tuple[float, float],
    direction: str,
    relative_range_change: float = 0.3,
    min_change: float = 0.05,
    max_change: float = 0.5,
) -> Tuple[float, float]:
    """Determines a new voltage range, to be measured in a subsequent tuning
    stage iteration. The range is shifted to more positive or negative values
    depending on ``direction``.

    Args:
        current_valid_ranges: Current voltage ranges.
        safety_voltage_ranges: List of safety ranges.
        direction: Range update directive, either 'positive' or 'negative'.
        relative_range_change: Relative voltage shift, in percent of current
            valid ranges.
        min_change: Minimum range change in Volt. New ranges will be shifted by
            at least this voltage difference.
        max_change: Maximum range change in Volt. New ranges will be shifted by
            at most this voltage difference.

    Returns:
        tuple: New voltage range.
    """

    new_min, new_max = current_valid_range
    if direction == "negative":
        diff = abs(current_valid_range[0] - safety_voltage_ranges[0])
        diff *= relative_range_change
        diff = min(diff, max_change)
        diff = max(diff, min_change)
        new_min, new_max = new_min - diff, new_max - diff

    elif direction == "positive":
        diff = abs(current_valid_range[1] - safety_voltage_ranges[1])
        diff *= relative_range_change
        diff = min(diff, max_change)
        diff = max(diff, min_change)
        new_min, new_max = new_min + diff, new_max + diff
    else:
        raise NotImplementedError("Unknown range update direction.")

    return (new_min, new_max)
