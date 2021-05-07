# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Any,
    Sequence,
)

import nanotune as nt
from nanotune.fit.dotfit import DotFit
from nanotune.classification.classifier import Classifier
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
        db_name: Name of database where run_id is saved.
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
    classifiers: Dict[str, Classifier],
    segment_ids: List[int],
    segment_db_name: Optional[str] = None,
    segment_db_folder: Optional[str] = None,
)-> Dict[int, int]:
    """Classifies several charge diagrams, such as previously segmented data.

    Args:
        classifiers: Dict with dot classifiers, single and double dot quality
            and dotregime.
        segment_ids: QCoDeS data run IDs.
        segment_db_name: Name of database where segment_ids are saved.
        segment_db_folder: Path to folder containing segment_db_name.

    Returns:
        dict: Mapping run IDs onto their classification result.
    """

    with nt.switch_database(segment_db_name, segment_db_folder):
        segment_regimes = {}
        clf_result = {}
        for data_id in segment_ids:
            for clf_type in ['singledot', 'doubledot', 'dotregime']:
                clf_result[clf_type] = classifiers[clf_type].predict(
                    data_id,
                    segment_db_name,
                    db_folder=segment_db_folder,
                )

            regime = resolve_dot_regime(clf_result)
            segment_regimes[data_id] = regime

    return segment_regimes


def resolve_dot_regime(
    classification_result: Dict[str, List[bool]],
) -> int:
    """

    """

    good_single = any(classification_result['singledot'])
    good_double = any(classification_result['doubledot'])
    dotregime = any(classification_result['dotregime'])

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
        raise ValueError('Unable to resolve dot regime.')

    return regime


def translate_regime(
    regime: int,
) -> Tuple[str, bool]:
    """ """

    rev_mapping = {}
    for str_regime in ['singledot', 'doubledot']:
        idx = DOT_LABEL_MAPPING[str_regime]
        rev_mapping[idx[0]] = (str_regime, False)
        rev_mapping[idx[1]] = (str_regime, True)

    return rev_mapping[regime]


def verify_dot_classification(
    target_regime: str,
    regimes: Sequence[int],
) -> bool:
    """ """

    good_found = False
    target = DOT_LABEL_MAPPING[target_regime][1]
    if target in regimes:
        good_found = True
    return good_found


def conclude_dot_classification(
    target_regime: str,
    dot_segments: Dict[int, Any],
) -> Tuple[str, bool]:
    """It expects the ``ml_result`` input to contain a 'dot_segments' entry,
    which containts the predicted regime of each 2D segment of the original
    measurement. The method uses ``verify_dot_classification`` to check
    whether the target regime has been found and ``translate_regime`` to
    interpret the classification result (int) in order to return a string
    and bool indicating the charge state and quality.

    Args:
        dot_segments= {
                <run_id>: {
                    'predicted_regime': int,
                    }
                }

    Returns:
        str: Charge state, e.g. 'singledot'
        bool: quality
    """

    dot_segment_vals = dot_segments.values()
    regimes = [seg['predicted_regime'] for seg in dot_segment_vals]
    good_found = verify_dot_classification(target_regime, regimes)

    if good_found:
        regime = target_regime
        quality = True
    else:
        majority_vote = max(regimes, key=regimes.count)
        regime, quality = translate_regime(majority_vote)

    return regime, quality


def get_range_directives_chargediagram(
    fit_range_update_directives: List[str],
    current_valid_ranges: List[Tuple[float, float]],
    safety_voltage_ranges: List[Tuple[float, float]],
) -> Tuple[List[str], List[str]]:
    """ """

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
    """ """

    if range_change_settings is None:
        range_change_settings = {
            'relative_range_change': 0.3,
            'min_change': 0.05,
            'max_change': 0.5,
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
                ('ChargeDiagram: Unknown action.' \
                 'Cannot update measurement setting')
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
            ('ChargeDiagram: Unknown range update directive.' \
             'Cannot update measurement setting')
        )

    return new_ranges

def get_new_range(
    current_valid_range: Tuple[float, float],
    safety_range: Tuple[float, float],
    direction: str,
    relative_range_change: float = 0.3,
    min_change: float = 0.05,
    max_change: float = 0.5,
) -> Tuple[float, float]:
    """ """

    new_min, new_max = current_valid_range
    if direction == "negative":
        diff = range_change * abs(current_valid_range[0] - safety_range[0])
        diff = min(diff, max_change)
        diff = max(diff, min_change)
        new_min = new_min - diff

    elif direction == "positive":
        diff = range_change * abs(current_valid_range[1] - safety_range[1])
        diff = min(diff, max_change)
        diff = max(diff, min_change)
        new_max = new_max + diff
    else:
        raise NotImplementedError("Unknown range update direction.")

    return (new_min, new_max)