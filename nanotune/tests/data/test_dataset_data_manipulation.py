import pytest
import pprint

import numpy as np
import scipy.signal as sg
import scipy.fftpack as fp
from skimage.transform import resize
from scipy.ndimage import sobel, generic_gradient_magnitude

import nanotune as nt
from nanotune.data.dataset import Dataset, default_readout_methods, default_coord_names
from nanotune.tests.data_generator_methods import generate_doubledot_data
from nanotune.math.gaussians import gaussian2D_fct

pp = pprint.PrettyPrinter(indent=4)


def test_dataset_1ddata_loading(nt_dataset_pinchoff, tmp_path):
    ds = Dataset(1, db_name="temp.db", db_folder=str(tmp_path))

    assert ds.exp_id == 1
    assert ds.dimensions["dc_current"] == 1
    assert ds.dimensions["dc_sensor"] == 1

    assert len(ds.raw_data) == 2
    assert len(ds.data) == 2

    vx = np.linspace(-0.1, 0, 120)
    ds_vx = ds.raw_data["current"]["voltage"].values
    assert np.allclose(ds_vx, vx)

    ds_sig = ds.raw_data["current"].values
    sig = 0.6 * (1 + np.tanh(1000 * vx + 50))
    assert np.allclose(ds_sig, sig)

    assert ds.get_plot_label("dc_current", 0) == "voltage x [V]"
    assert ds.get_plot_label("dc_sensor", 0) == "voltage x [V]"
    assert ds.get_plot_label("dc_current", 1) == "dc current [A]"
    assert ds.get_plot_label("dc_sensor", 1) == "dc sensor [A]"
    with pytest.raises(AssertionError):
        ds.get_plot_label("dc_sensor", 2)


# TODO: check raw_data to data conversion


def test_dataset_2ddata_loading(nt_dataset_doubledot, tmp_path):
    ds = Dataset(1, db_name="temp.db", db_folder=str(tmp_path))

    assert ds.exp_id == 1
    assert ds.dimensions["dc_current"] == 2
    assert ds.dimensions["dc_sensor"] == 2
    assert len(ds.raw_data) == 2
    assert len(ds.data) == 2

    ds_vx = ds.raw_data["current"]["v_x"].values
    ds_vy = ds.raw_data["current"]["v_y"].values
    ds_sig = ds.raw_data["current"].values
    ds_sens = ds.raw_data["sensor"].values

    xv, yv, ddot, sensor = generate_doubledot_data()
    x = np.unique(xv)
    y = np.unique(yv)

    assert np.allclose(ds_vx, x)
    assert np.allclose(ds_vy, y)
    assert np.allclose(ds_sig, ddot.T)
    assert np.allclose(ds_sens, sensor.T)

    assert ds.get_plot_label("dc_current", 0) == "voltage x [V]"
    assert ds.get_plot_label("dc_sensor", 0) == "voltage x [V]"
    assert ds.get_plot_label("dc_current", 1) == "voltage y [V]"
    assert ds.get_plot_label("dc_sensor", 1) == "voltage y [V]"
    assert ds.get_plot_label("dc_current", 2) == "dc current [A]"
    assert ds.get_plot_label("dc_sensor", 2) == "dc sensor [A]"


def test_dataset_normalisation(nt_dataset_pinchoff, tmp_path):
    ds = Dataset(1, db_name="temp.db", db_folder=str(tmp_path))

    v_x = np.linspace(-0.1, 0, 100)
    sig = 0.6 * (1 + np.tanh(1000 * v_x + 50))

    norm_sig = ds._normalize_data(sig, "dc_current")
    manually_normalized = sig / 1.2

    assert np.allclose(manually_normalized, norm_sig)
    assert np.max(norm_sig) <= 1.0
    assert np.min(norm_sig) >= 0.0


# TODO: def test_dataset_missing_data()
# if np.isnan(np.sum(signal)):
#     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#     signal = imp.fit_transform(signal)


def test_dataset_1d_frequencies(nt_dataset_pinchoff, tmp_path):
    ds = Dataset(1, db_name="temp.db", db_folder=str(tmp_path))

    ds.compute_power_spectrum()
    assert len(ds.power_spectrum) == 2

    assert len(ds.power_spectrum) == len(ds.data)
    with pytest.raises(KeyError):
        ds.power_spectrum["freq_y"]

    ds_vx = ds.data["dc_current"][default_coord_names["voltage"][0]].values
    ds_sig = ds.data["dc_current"].values

    xv = np.unique(ds_vx)
    signal = ds_sig.copy()
    signal = sg.detrend(signal, axis=0)

    frequencies_res = fp.fft(signal)
    frequencies_res = np.abs(fp.fftshift(frequencies_res)) ** 2

    fx = fp.fftshift(fp.fftfreq(frequencies_res.shape[0], d=xv[1] - xv[0]))
    coord_name = default_coord_names["frequency"][0]
    ds_fx = ds.power_spectrum["dc_current"][coord_name].values
    ds_freq = ds.power_spectrum["dc_current"].values

    assert np.allclose(ds_fx, fx)
    assert np.allclose(ds_freq, frequencies_res)


def test_dataset_2d_frequencies(nt_dataset_doubledot, tmp_path):
    ds = Dataset(1, db_name="temp.db", db_folder=str(tmp_path))

    ds.compute_power_spectrum()
    assert len(ds.power_spectrum) == 2

    ds_vx = ds.data["dc_current"][default_coord_names["voltage"][0]].values
    ds_vy = ds.data["dc_current"][default_coord_names["voltage"][1]].values
    ds_curr = ds.data["dc_current"].values.copy()

    xv = np.unique(ds_vx)
    yv = np.unique(ds_vy)

    ds_curr = sg.detrend(ds_curr, axis=0)
    ds_curr = sg.detrend(ds_curr, axis=1)

    frequencies_res = fp.fft2(ds_curr)
    frequencies_res = np.abs(fp.fftshift(frequencies_res)) ** 2

    fx_1d = fp.fftshift(fp.fftfreq(frequencies_res.shape[0], d=xv[1] - xv[0]))
    fy_1d = fp.fftshift(fp.fftfreq(frequencies_res.shape[1], d=yv[1] - yv[0]))

    # fx, fy = np.meshgrid(fx_1d, fy_1d, indexing="ij")
    # frequencies_res = np.abs(frequencies_res)
    coord_name = default_coord_names["frequency"][0]
    ds_fx = ds.power_spectrum["dc_current"][coord_name].values
    coord_name = default_coord_names["frequency"][1]
    ds_fy = ds.power_spectrum["dc_current"][coord_name].values
    ds_freq = ds.power_spectrum["dc_current"].values

    assert np.allclose(ds_fx, fx_1d)
    assert np.allclose(ds_fy, fy_1d)
    assert np.allclose(ds_freq, frequencies_res)


def test_1D_prepare_filtered_data(nt_dataset_pinchoff, tmp_path):
    pf = Dataset(1, "temp.db", db_folder=str(tmp_path))
    pf.prepare_filtered_data()

    assert len(pf.filtered_data) == len(pf.data)
    assert pf.filtered_data.dc_current.shape == pf.data.dc_current.shape
    rtol = 1e-05
    assert not np.allclose(
        pf.filtered_data.dc_sensor.values, pf.data.dc_sensor.values, rtol=rtol
    )
