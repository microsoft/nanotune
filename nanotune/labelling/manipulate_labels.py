import os
import logging
from typing import Optional
import qcodes as qc
from qcodes.dataset.experiment_container import load_by_id
from qcodes.dataset.plotting import plot_by_id

import nanotune as nt

logger = logging.getLogger(__name__)
LABELS = list(dict(nt.config["core"]["labels"]).keys())


def correct_label(
    dataid: int,
    db_name: str,
    new_stage: str,
    new_quality: int,
    db_folder: Optional[str] = None,
) -> bool:
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    if db_name[-2:] != "db":
        db_name += ".db"
    db_folder = os.path.join(nt.config["db_folder"], db_name)
    qc.config["core"]["db_location"] = db_folder

    ds = load_by_id(dataid)

    if new_stage not in LABELS:
        logger.error("Wrong tuning stage. Leaving old label.")
        return False
    else:
        new_label = dict.fromkeys(LABELS, 0)
        new_label[new_stage] = 1
        new_label["good"] = int(new_quality)

        for label, value in new_label.items():
            ds.add_metadata(label, value)

    return True


def print_label(
    dataid: int,
    db_name: str,
    db_folder: Optional[str] = None,
    plot_data: Optional[bool] = True,
) -> None:
    """"""
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    ds = load_by_id(dataid)
    if plot_data:
        plot_by_id(dataid)
    print("dataset {} in {}: ".format(dataid, db_name))
    quality_mapping = {1: "good", 0: "poor"}
    for label in LABELS:
        if label != "good":
            if int(ds.get_metadata(label)) == 1:
                quality = quality_mapping[int(ds.get_metadata("good"))]
                print("{} {}.".format(quality, label))
