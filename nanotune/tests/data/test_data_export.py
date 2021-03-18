import os
import pytest

import scipy.fft as fp

from skimage.transform import resize
from scipy.ndimage import sobel, generic_gradient_magnitude

import numpy as np
import qcodes as qc
from qcodes.dataset.sqlite.queries import (
    new_experiment as ne,
    finish_experiment,
    get_run_counter,
    get_runs,
    get_last_run,
    get_last_experiment,
    get_experiments,
)
from qcodes.dataset.sqlite.database import (
    get_DB_location,
    get_DB_debug,
    connect,
    conn_from_dbpath_or_conn,
)

from qcodes.dataset.experiment_container import load_last_experiment, experiments
import nanotune as nt
from nanotune.data.export_data import (
    export_label,
    export_data,
    correct_normalizations,
    # subsample_2Ddata,
)


def test_export_label():
    new_label = export_label(["singledot"], 0, "dotregime")
    assert new_label == 0

    new_label = export_label(["singledot"], 1, "dotregime")
    assert new_label == 1

    new_label = export_label(["doubledot"], 0, "dotregime")
    assert new_label == 2

    new_label = export_label(["doubledot"], 1, "dotregime")
    assert new_label == 3

    for category in ["outerbarriers", "pinchoff", "singledot", "doubledot"]:
        for quality in [0, 1]:
            new_label = export_label([category], quality, category)
            assert new_label == quality

    with pytest.raises(ValueError):
        new_label = export_label(["pinchoff"], quality, "just work")
    with pytest.raises(ValueError):
        new_label = export_label(["outerbarriers"], quality, "pinchoff")


def test_export_data(experiment_doubledots, tmp_path):

    export_data(
        "doubledot", ["temp.db"], ["doubledot"], db_folder=tmp_path, filename="temp.npy"
    )
    data_w_labels = np.load(os.path.join(tmp_path, "temp.npy"))
    assert data_w_labels.shape == (4, 10, 2501)

    export_data(
        "doubledot",
        ["temp.db"],
        ["doubledot"],
        skip_ids={"temp.db": [1, 10]},
        db_folder=tmp_path,
        filename="temp.npy",
    )
    data_w_labels = np.load(os.path.join(tmp_path, "temp.npy"))
    assert data_w_labels.shape == (4, 8, 2501)

    export_data(
        "doubledot", ["temp.db"], ["singledot"], db_folder=tmp_path, filename="temp.npy"
    )
    data_w_labels = np.load(os.path.join(tmp_path, "temp.npy"))
    assert data_w_labels.shape == (4, 0, 2501)

    export_data(
        "doubledot",
        ["temp.db"],
        ["doubledot"],
        quality=1,
        db_folder=tmp_path,
        filename="temp.npy",
    )
    data_w_labels = np.load(os.path.join(tmp_path, "temp.npy"))
    assert data_w_labels.shape == (4, 10, 2501)


def test_correct_normalizations(experiment_doubledots, tmp_path):
    export_data(
        "doubledot", ["temp.db"], ["doubledot"], db_folder=tmp_path, filename="temp.npy"
    )

    data_w_labels = np.load(os.path.join(tmp_path, "temp.npy"))
    sg_indx = nt.config["core"]["data_types"]["signal"]

    data_w_labels[sg_indx, :, :-1] += 1
    assert np.max(data_w_labels[sg_indx, :, :-1].flatten()) >= 1

    path = os.path.join(tmp_path, "temp.npy")
    np.save(path, data_w_labels)

    correct_normalizations("temp.npy", tmp_path)

    data_w_labels = np.load(os.path.join(tmp_path, "temp.npy"))
    assert data_w_labels.shape == (4, 10, 2501)

    assert np.max(data_w_labels[sg_indx, :, :-1].flatten()) <= 1
    assert np.min(data_w_labels[sg_indx, :, :-1].flatten()) >= 0

    data = data_w_labels[:, :, :-1]

    for did, signal in enumerate(data[sg_indx]):
        freq_spect = fp.fft2(signal.reshape(50, 50))
        freq_spect = np.abs(fp.fftshift(freq_spect))

        grad = generic_gradient_magnitude(signal.reshape(50, 50), sobel)

        index = nt.config["core"]["data_types"]["frequencies"]
        assert np.allclose(data[index, did, :], freq_spect.flatten())

        index = nt.config["core"]["data_types"]["gradient"]
        assert np.allclose(data[index, did, :], grad.flatten())
