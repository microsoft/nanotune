import glob
import os
import ntpath
import time
import logging
from contextlib import contextmanager

from typing import Optional, Dict, Any, List, Tuple
import qcodes as qc
from qcodes.dataset.experiment_container import experiments
from qcodes.dataset.sqlite.database import connect
from qcodes.dataset.sqlite.connection import atomic
from qcodes.dataset.sqlite.query_helpers import many_many, insert_column
from qcodes.dataset.sqlite.queries import add_meta_data, get_metadata


import nanotune as nt
from nanotune.data.dataset import Dataset
from nanotune.utils import flatten_list

logger = logging.getLogger(__name__)


def get_dataIDs(
    db_name: str,
    stage: str,
    db_folder: Optional[str] = None,
    quality: Optional[int] = None,
) -> List[int]:
    """"""
    if db_name[-2:] != "db":
        db_name += ".db"
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    db_path = os.path.join(db_folder, db_name)
    conn = connect(db_path)

    if quality is None:
        sql = f"""
            SELECT run_id FROM runs WHERE {stage}={1} OR {stage} LIKE {str(1)}
            """
    else:
        sql = f"""
            SELECT run_id FROM runs
                WHERE ({stage}={1} OR {stage} LIKE {str(1)})
                AND (good={quality} OR good LIKE {str(quality)})
            """
    c = conn.execute(sql)
    param_names_temp = many_many(c, "run_id")

    return list(flatten_list(param_names_temp))


def get_unlabelled_ids(
    db_name: str,
    db_folder: Optional[str] = None,
) -> List[int]:
    """"""
    if db_name[-2:] != "db":
        db_name += ".db"
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    db_path = os.path.join(db_folder, db_name)

    conn = connect(db_path)
    sql = f"""
        SELECT run_id FROM runs WHERE good IS NULL
        """
    c = conn.execute(sql)
    param_names_temp = many_many(c, "run_id")

    return list(flatten_list(param_names_temp))


def list_experiments(
    db_folder: Optional[str] = None,
) -> Tuple[str, Dict[str, List[int]]]:
    """"""
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    # print(os.listdir(db_folder))
    # print(db_folder)
    # db_files = glob.glob(db_folder + '*.db')
    # db_files = glob.glob(db_folder)
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
    """
    Ceate new database and initialise it
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
    """"""
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
    """"""
    db_path = qc.config["core"]["db_location"]
    db_name = os.path.basename(db_path)
    db_folder = os.path.dirname(db_path)

    return db_name, db_folder


def get_last_dataid(
    db_name: str,
    db_folder: Optional[str] = None,
) -> int:
    """
    Return last 'global' dataid for given database. It is this ID that is used
    in plot_by_id
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
    """ """
    original_db, original_db_folder = nt.get_database()
    nt.set_database(temp_db_name, db_folder=temp_db_folder)
    try:
        yield
    finally:
        nt.set_database(original_db, db_folder=original_db_folder)


# def rename_labels(db_name: str,
#                   db_folder: Optional[str] = nt.config['db_folder']) -> bool:
#     """
#     """
#     nt.set_database(db_name)
#     for data_id in range(1, nt.get_last_dataid(db_name)+1):
#         print(data_id)
#         nt.correct_label_names(data_id, db_name)
#         time.sleep(0.1)

# if sqlite3.sqlite_version is 3.25 and above, we can use ALTER TABLE to
# rename columns:
# conn = connect(db_path)
# sql = f"""
#     ALTER TABLE runs RENAME COLUMN wallwall TO leadscoupling;
#     """
# c = conn.execute(sql)

# ALTER TABLE runs RENAME COLUMN wallwall TO leadcoupling;
# ALTER TABLE runs RENAME COLUMN clmboscs TO coulomboscillations;
# ALTER TABLE runs RENAME COLUMN clmbdiam TO coulombdiamonds;
# ALTER TABLE runs RENAME COLUMN zbp TO zerobiaspeak;
