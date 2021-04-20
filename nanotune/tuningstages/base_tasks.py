# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import json
from qcodes.dataset.experiment_container import load_by_id
import nanotune as nt
from nanotune.classification.classifier import Classifier


def save_predicted_category(
    run_id: int,
    predicted_quality: int,
) -> None:
    """"""
    ds = load_by_id(run_id)
    nt_meta = json.loads(ds.get_metadata(nt.meta_tag))

    nt_meta["predicted_category"] = predicted_quality
    ds.add_metadata(nt.meta_tag, json.dumps(nt_meta))


def check_quality(
    classifier: Classifier,
    run_id: int,
    db_name: str,
    db_folder: str,
) -> bool:
    quality = clf.predict(run_id, db_name, db_folder)
    return any(quality)