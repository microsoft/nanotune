import logging
import ntpath
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import qcodes as qc
from qcodes.dataset.experiment_container import experiments
from qcodes.dataset.sqlite.connection import atomic
from qcodes.dataset.sqlite.database import connect
from qcodes.dataset.sqlite.queries import add_meta_data, get_metadata
from qcodes.dataset.sqlite.query_helpers import many_many

import nanotune as nt
from nanotune.utils import flatten_list

logger = logging.getLogger(__name__)


def get_dataIDs(
    db_name: str,
    category: str,
    db_folder: Optional[str] = None,
    quality: Optional[int] = None,
) -> List[int]:
    """Returns QCoDeS run IDs of datasets belonging to a specific category
    of measurements, assigned to the data during labelling.

    The entire database `db_name` is searched and a column of the same name as
    the category looked up needs to exist in it. Example of categories are
    `pinchoff`, `singledot` or `doubledot`, all valid ones are defined in
    configuration.config.json under the "labels" key.
    Optionally, the quality can be
    specified. Quality and category together are the machine learning labels
    nanotune uses during classification.

    Args:
        db_name: name of database to search.
        category: measurement type/category we are looking for. E.g.
            `pinchoff`, `singledot` or `doubledot`.
        db_folder: folder containing database. If not specified,
            `nt.config["db_folder"]` is used.
        quality: Optional if a specific quality is required. 1==good, 0==poor.

    Returns:
        List of QCoDeS run IDs.
    """
    if db_name[-2:] != "db":
        db_name += ".db"
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    db_path = os.path.join(db_folder, db_name)
    conn = connect(db_path)

    if quality is None:
        sql = f"""
            SELECT run_id FROM runs WHERE {category}={1} OR {category} LIKE {str(1)}
            """
    else:
        sql = f"""
            SELECT run_id FROM runs
                WHERE ({category}={1} OR {category} LIKE {str(1)})
                AND (good={quality} OR good LIKE {str(quality)})
            """
    c = conn.execute(sql)
    param_names_temp = many_many(c, "run_id")

    return list(flatten_list(param_names_temp))


def get_unlabelled_ids(
    db_name: str,
    db_folder: Optional[str] = None,
    get_run_id: bool = True,
) -> List[int]:
    """Gets run IDs all unlabelled datasets in a database. A dataset is selected
    if it doesn't have a value in the `good` column.

    Args:
        db_name: name of database to search.
        db_folder: folder containing database. If not specified,
            `nt.config["db_folder"]` is used.
        get_run_id: whether to return run IDs. Returns capturd run IDs if False.

    Returns:
        List of QCoDeS run IDs which do not have a machine learning label.
    """
    if db_name[-2:] != "db":
        db_name += ".db"
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    db_path = os.path.join(db_folder, db_name)
    conn = connect(db_path)
    if get_run_id:
        sql = f"""
            SELECT run_id FROM runs WHERE good IS NULL
            """
    else:
        sql = f"""
            SELECT captured_run_id FROM runs WHERE good IS NULL
            """
    c = conn.execute(sql)
    if get_run_id:
        param_names_temp = many_many(c, "run_id")
    else:
        param_names_temp = many_many(c, "captured_run_id")

    return list(flatten_list(param_names_temp))


def list_experiments(
    db_folder: Optional[str] = None,
) -> Tuple[str, Dict[str, List[int]]]:
    """Lists all databases and their experiments within a folder.

    Args:
        db_folder: the folder to enumerate.

    Returns:
        str: name of folder which was searched.
        dict: mapping database name to a list of experiment IDs.
    """
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    all_fls = os.listdir(db_folder)
    db_files = [db_file for db_file in all_fls if db_file.endswith(".db")]
    print("db_files: {}".format(db_files))
    db_names = [ntpath.basename(path) for path in db_files]

    all_experiments = {}
    for idb, db_file in enumerate(db_files):
        qc.config["core"]["db_location"] = os.path.join(db_folder, db_file)

        exp_ids = []
        exps = experiments()
        for exprmt in exps:
            exp_ids.append(exprmt.exp_id)

        all_experiments[db_names[idb]] = exp_ids

    return db_folder, all_experiments


def new_database(
    db_name: str,
    db_folder: Optional[str] = None,
) -> str:
    """Create new database and initialise it with nanotune labels.
    A separate column for each label is created.

    Args:
        db_name: name of new database.
        db_folder: folder containing database. If not specified,
            `nt.config["db_folder"]` is used.

    Returns:
        str: absolute path of new database.
    """
    if db_folder is None:
        db_folder = nt.config["db_folder"]
    else:
        nt.config["db_folder"] = db_folder

    if db_name[-2:] != "db":
        db_name += ".db"
    path = os.path.join(db_folder, db_name)
    qc.initialise_or_create_database_at(path)
    nt.config["db_name"] = db_name

    # add label columns
    db_conn = connect(path)
    with atomic(db_conn) as conn:
        add_meta_data(conn, 0, {"original_guid": 0})
        for label in nt.config["core"]["labels"]:
            add_meta_data(conn, 0, {label: 0})

    return path


def set_database(
    db_name: str,
    db_folder: Optional[str] = None,
) -> None:
    """Sets a new database to be the default one to load from and save to.
    If the database does not exist, a new one is created. The change is
    propagated to QCoDeS' configuration.

    Args:
        db_name: name of database to set.
        db_folder: folder containing database. If not specified,
            `nt.config["db_folder"]` is used.
    """
    if db_folder is None:
        db_folder = nt.config["db_folder"]
    else:
        nt.config["db_folder"] = db_folder

    if db_name[-2:] != "db":
        db_name += ".db"
    nt.config["db_name"] = db_name
    db_path = os.path.join(db_folder, db_name)
    if not os.path.isfile(db_path):
        nt.new_database(db_name, db_folder)

    qc.config["core"]["db_location"] = db_path

    # check if label columns exist, create if not
    db_conn = connect(db_path)
    with atomic(db_conn) as conn:
        try:
            get_metadata(db_conn, "0", nt.config["core"]["labels"][0])
        except (RuntimeError, KeyError):
            for label in nt.config["core"]["labels"]:
                add_meta_data(conn, 0, {label: 0})


def get_database() -> Tuple[str, str]:
    """Gets name and folder of the current default database used by QCoDeS.

    Returns:
        str: database name
        str: folder containing database
    """
    db_path = qc.config["core"]["db_location"]
    db_name = os.path.basename(db_path)
    db_folder = os.path.dirname(db_path)

    return db_name, db_folder


def get_last_dataid(
    db_name: str,
    db_folder: Optional[str] = None,
) -> int:
    """Returns QCoDeS run ID of last dataset in a database, i.e the highest
    run ID there is.

    Args:
        db_name: name of database.
        db_folder: folder containing database. If not specified,
            `nt.config["db_folder"]` is used.

    Returns:
        int: sum of `last_counter` of all experiments within a database.
    """
    if db_folder is None:
        db_folder = nt.config["db_folder"]
    if db_name[-2:] != "db":
        db_name += ".db"

    nt.set_database(db_name, db_folder=db_folder)
    last_index = 0
    for experiment in experiments():
        last_index += experiment.last_counter

    return last_index


@contextmanager
def switch_database(temp_db_name: str, temp_db_folder: str):
    """Context manager temporarily setting a different database. Sets back to
    previous database name and folder.

    Args:
        temp_db_name: name of database to set.
        temp_db_folder: folder containing database. If not specified,
            `nt.config["db_folder"]` is used.
    """
    original_db, original_db_folder = nt.get_database()
    nt.set_database(temp_db_name, db_folder=temp_db_folder)
    try:
        yield
    finally:
        nt.set_database(original_db, db_folder=original_db_folder)
