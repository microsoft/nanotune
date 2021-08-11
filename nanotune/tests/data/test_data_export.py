import os

import numpy as np
import pytest
import scipy.fft as fp
from scipy.ndimage import generic_gradient_magnitude, sobel
from skimage.transform import resize

import nanotune as nt
from nanotune.data.export_data import (
    correct_normalizations, export_data, export_label, prep_data)


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
        "doubledot", ["temp.db"], db_folder=tmp_path, filename="temp.npy"
    )
    data_w_labels = np.load(os.path.join(tmp_path, "temp.npy"))
    assert data_w_labels.shape == (4, 10, 2501)

    export_data(
        "doubledot",
        ["temp.db"],
        skip_ids={"temp.db": [1, 10]},
        db_folder=tmp_path,
        filename="temp.npy",
    )
    data_w_labels = np.load(os.path.join(tmp_path, "temp.npy"))
    assert data_w_labels.shape == (4, 8, 2501)

    export_data(
        "doubledot",
        ["temp.db"],
        quality=1,
        db_folder=tmp_path,
        filename="temp.npy",
    )
    data_w_labels = np.load(os.path.join(tmp_path, "temp.npy"))
    assert data_w_labels.shape == (4, 10, 2501)


def test_correct_normalizations(experiment_doubledots, tmp_path):
    export_data(
        "doubledot", ["temp.db"], db_folder=tmp_path, filename="temp.npy"
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


def test_prep_data_return_shape(nt_dataset_pinchoff, tmp_path):
    ds = nt.Dataset(1, db_name="temp.db", db_folder=str(tmp_path))
    shape = tuple(nt.config["core"]["standard_shapes"]["1"])

    condensed_data = np.array(prep_data(ds, "pinchoff")[0])

    assert len(ds.power_spectrum) > 0
    index = nt.config["core"]["data_types"]["signal"]

    assert len(condensed_data[index, 0, :]) == np.prod(shape)

    index = nt.config["core"]["data_types"]["frequencies"]
    assert len(condensed_data[index, 0, :]) == np.prod(shape)

    index = nt.config["core"]["data_types"]["gradient"]
    assert len(condensed_data[index, 0, :]) == np.prod(shape)

    index = nt.config["core"]["data_types"]["features"]
    assert len(condensed_data[index, 0, :]) == np.prod(shape)

    with pytest.raises(KeyError):
        condensed_data = prep_data(ds, "doubledot")[0]


def test_prep_data_normalization(nt_dataset_pinchoff, tmp_path):
    ds = nt.Dataset(1, db_name="temp.db", db_folder=str(tmp_path))

    ds.data["transport"].values *= 1.4
    ds.data["transport"].values += 0.5
    assert np.max(ds.data["transport"].values) > 1
    assert np.min(ds.data["transport"].values) >= 0.5

    _ = prep_data(ds, "pinchoff")[0]

    assert np.max(ds.data["transport"].values) <= 1
    assert np.min(ds.data["transport"].values) <= 0.5


def test_prep_data_return_data(nt_dataset_pinchoff, tmp_path):
    ds = nt.Dataset(1, db_name="temp.db", db_folder=str(tmp_path))
    condensed_data = np.array(prep_data(ds, "pinchoff")[0])

    shape = tuple(nt.config["core"]["standard_shapes"]["1"])

    ds_curr = ds.data["transport"].values
    ds_freq = ds.power_spectrum["transport"].values

    data_resized = resize(ds_curr, shape, anti_aliasing=True, mode="constant").flatten()
    frq = resize(ds_freq, shape, anti_aliasing=True, mode="constant").flatten()

    grad = generic_gradient_magnitude(ds_curr, sobel)
    gradient_resized = resize(
        grad, shape, anti_aliasing=True, mode="constant"
    ).flatten()

    relevant_features = nt.config["core"]["features"]["pinchoff"]
    features = []
    for feat in relevant_features:
        features.append(ds.features["transport"][feat])

    pad_width = len(data_resized.flatten()) - len(features)
    features = np.pad(
        features,
        (0, pad_width),
        "constant",
        constant_values=nt.config["core"]["fill_value"],
    )

    index = nt.config["core"]["data_types"]["signal"]
    assert np.allclose(condensed_data[index, 0, :], data_resized)

    index = nt.config["core"]["data_types"]["frequencies"]
    assert np.allclose(condensed_data[index, 0, :], frq)

    index = nt.config["core"]["data_types"]["gradient"]
    assert np.allclose(condensed_data[index, 0, :], gradient_resized)

    index = nt.config["core"]["data_types"]["features"]
    assert np.allclose(condensed_data[index, 0, :], features)
