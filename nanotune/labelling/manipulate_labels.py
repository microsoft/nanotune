import logging
from typing import Optional, Union

import qcodes as qc
from qcodes.dataset.plotting import plot_by_id

import nanotune as nt

logger = logging.getLogger(__name__)
LABELS = list(dict(nt.config["core"]["labels"]).keys())


def correct_label(
    dataid: int,
    db_name: str,
    new_stage: str,
    new_quality: Union[int, bool],
    db_folder: Optional[str] = None,
):
    """Sets the (nanotune) label of a dataset.

    Args:
        dataid: QCoDeS (captured) run ID
        db_name: database name
        new_stage: string indicating the new stage, e.g. "pinchoff". Valid
            stages are defined in config.py under the `labels` key.
        new_quality: quality of the dataset, 1==good, 0==poor.
        db_folder: path to folder containing database
    """
    nt.set_database(db_name, db_folder=db_folder)

    ds = qc.load_by_run_spec(captured_run_id=dataid)

    if new_stage not in LABELS:
        raise ValueError("Wrong tuning stage. Leaving old label.")
    else:
        new_label = dict.fromkeys(LABELS, 0)
        new_label[new_stage] = 1
        new_label["good"] = int(new_quality)

        for label, value in new_label.items():
            ds.add_metadata(label, value)
        logger.info('Label corrected successfully.')


def print_label(
    dataid: int,
    db_name: str,
    db_folder: Optional[str] = None,
    plot_data: Optional[bool] = True,
) -> None:
    """Prints nanotune label assigned to a dataset.

    Args:
        dataid: QCoDeS (captured) run ID
        db_name: database name
        db_folder: path to folder containing database
        plot_data: whether the dataset should also be plotted.
    """
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    nt.set_database(db_name, db_folder=db_folder)
    ds = qc.load_by_run_spec(captured_run_id=dataid)
    if plot_data:
        plot_by_id(dataid)
    msg = "Dataset {} in {}: ".format(dataid, db_name)
    print(msg)
    logger.info(msg)

    labelled_quality = ds.get_metadata("good")
    if labelled_quality is None:
        msg = 'has not been labelled yet.'
        print(msg)
        logger.info(msg)
    else:
        quality_mapping = {1: "good", 0: "poor"}
        quality = quality_mapping[int(ds.get_metadata("good"))]

        for label in LABELS:
            if label != "good":
                if int(ds.get_metadata(label)) == 1:
                    quality = quality_mapping[int(ds.get_metadata("good"))]
                    msg = "{} {}.".format(quality, label)
                    print(msg)
                    logger.info(msg)
