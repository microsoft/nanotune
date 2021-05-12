import gc
import os
import json
import pytest
import qcodes as qc
from qcodes.dataset.sqlite.database import connect, initialise_database
from qcodes import new_data_set, new_experiment
from qcodes.dataset.experiment_container import experiments, load_by_id
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.measurements import Measurement

import nanotune as nt
from nanotune.tests.data_generator_methods import (
    generate_doubledot_data,
    generate_default_metadata,
)
from nanotune.tests.data_savers import save_1Ddata_with_qcodes, save_2Ddata_with_qcodes

NT_LABELS = list(dict(nt.config["core"]["labels"]).keys())
META_FIELDS = nt.config["core"]["meta_fields"]
test_data_labels = {
    0: "pinchoff",
    1: "pinchoff",
    2: "singledot",
    3: None,
    4: "doubledot",
    5: "pinchoff",
    6: None,
    7: None,
    8: "doubledot",
    9: None,
}
test_data_labels2 = {
    0: "doubledot",
    1: "pinchoff",
    2: "singledot",
    3: None,
    4: "pinchoff",
    5: "singledot",
}


@pytest.fixture(scope="function")
def nt_dataset_doubledot_partial_metadata(experiment, tmp_path):
    datasaver = save_2Ddata_with_qcodes(generate_doubledot_data, None)

    current_label = {}
    current_label["doubledot"] = 1
    current_label["singledot"] = 0

    for label, value in current_label.items():
        datasaver.dataset.add_metadata(label, value)
    meta_dict = {
        "device_max_signal": 1,
        "normalization_constants": {"dc_current": [0, 1.4]},
    }
    datasaver.dataset.add_metadata(nt.meta_tag, json.dumps(meta_dict))
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function")
def experiment_labelled_data(empty_temp_db, tmp_path):
    e = new_experiment("test_experiment", sample_name="test_sample")
    for did in range(len(test_data_labels)):
        ds = DataSet(os.path.join(tmp_path, "temp.db"))

        nt_metadata, current_label = generate_default_metadata()
        stage = test_data_labels[did]
        if stage is not None:
            current_label[stage] = 1

        ds.add_metadata(nt.meta_tag, json.dumps(nt_metadata))
        ds.add_metadata("snapshot", json.dumps({}))
        for label, value in current_label.items():
            ds.add_metadata(label, value)
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope="function")
def experiment_partially_labelled(empty_temp_db, tmp_path):
    e = new_experiment("test_experiment", sample_name="test_sample")
    for did in range(len(test_data_labels)):
        ds = DataSet(os.path.join(tmp_path, "temp.db"))

        nt_metadata, current_label = generate_default_metadata()
        stage = test_data_labels[did]
        if stage is not None:
            current_label[stage] = 1
            for label, value in current_label.items():
                ds.add_metadata(label, value)

        ds.add_metadata(nt.meta_tag, json.dumps(nt_metadata))
        ds.add_metadata("snapshot", json.dumps({}))
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope="function")
def second_third_experiment_labelled_data(second_empty_temp_db, tmp_path):

    e1 = new_experiment("test_experiment2", sample_name="test_sample")
    e2 = new_experiment("test_experiment3", sample_name="test_sample")

    for did in range(len(test_data_labels2)):
        ds = DataSet(os.path.join(tmp_path, "temp2.db"), exp_id=e2._exp_id)

        nt_metadata, current_label = generate_default_metadata()
        stage = test_data_labels2[did]
        if stage is not None:
            current_label[stage] = 1

        ds.add_metadata(nt.meta_tag, json.dumps(nt_metadata))
        ds.add_metadata("snapshot", json.dumps({}))
        for label, value in current_label.items():
            ds.add_metadata(label, value)
    try:
        yield e2
    finally:
        e1.conn.close()
        e2.conn.close()
