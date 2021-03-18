import os
import pytest
import sqlite3

from qcodes import new_data_set, new_experiment
from qcodes.dataset.sqlite.database import connect
from qcodes.dataset.sqlite.connection import atomic

import nanotune as nt
from nanotune.data.databases import (
    get_dataIDs,
    get_unlabelled_ids,
    list_experiments,
    new_database,
    set_database,
    get_database,
    get_last_dataid,
)


def test_database_creation_and_init(tmp_path):

    db_folder = os.path.join(tmp_path, "temp.db")
    assert not os.path.exists(db_folder)

    nt.new_database("temp.db", str(tmp_path))
    assert os.path.exists(db_folder)

    # Make sure label columnbs are created. Following lines should not
    # raise an exception
    ids = get_dataIDs("temp.db", "pinchoff", db_folder=tmp_path)
    assert ids == []
    ids = get_dataIDs("temp.db", "good", db_folder=tmp_path)
    assert ids == []


def test_dataID_selection(experiment_labelled_data, tmp_path):

    ids = get_dataIDs("temp.db", "pinchoff", db_folder=tmp_path)
    assert ids == [1, 2, 6]

    ids = get_dataIDs("temp.db", "singledot", db_folder=tmp_path)
    assert ids == [3]

    with pytest.raises(sqlite3.OperationalError):
        ids = get_dataIDs("temp.db", "single", db_folder=tmp_path)


def test_unlabelled_ids(experiment_partially_labelled, tmp_path):
    ids = get_unlabelled_ids("temp.db", str(tmp_path))
    assert ids == [4, 7, 8, 10]


def test_list_experiments(
    experiment_labelled_data, second_third_experiment_labelled_data, tmp_path
):

    db_folder, all_experiments = list_experiments(str(tmp_path))

    print(all_experiments)

    assert db_folder == str(tmp_path)
    assert all_experiments["temp.db"] == [1]
    assert all_experiments["temp2.db"] == [1, 2]


def test_get_database(empty_temp_db, experiment_different_db_folder, tmp_path):
    set_database("temp.db", tmp_path)
    db_name, db_folder = get_database()
    assert db_name == "temp.db"
    assert db_folder == str(tmp_path)

    path2 = os.path.join(str(tmp_path), "test")
    set_database("temp2.db", path2)
    db_name, db_folder = get_database()
    assert db_name == "temp2.db"
    assert db_folder == path2


def test_get_last_dataid(experiment_labelled_data, tmp_path):
    last_id = get_last_dataid("temp.db", str(tmp_path))
    assert last_id == 10
