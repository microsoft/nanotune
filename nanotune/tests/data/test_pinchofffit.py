import os
import json
import copy
import math
import pytest
import numpy as np
from qcodes.dataset.experiment_container import load_by_id
import nanotune as nt
from nanotune.fit.pinchofffit import PinchoffFit
from nanotune.tests.data_generator_methods import generate_bad_pinchoff_data

rtol = 1e-05


def test_pinchofffit_init(nt_dataset_pinchoff, tmp_path):
    attributes = [
        "features",
        "gradient_percentile",
        "_low_signal",
        "_high_signal",
        "_transition_voltage",
        "_transition_signal",
        "_low_signal_index",
        "_high_signal_index",
        "_transition_signal_index",
        "_normalized_voltage",
        "_range_update_directives",
    ]

    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))

    for attr in attributes:
        assert hasattr(pf, attr)
        getattr(pf, attr)


def test_pinchofffit_range_update_directives(nt_dataset_pinchoff, tmp_path):
    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))

    assert not pf.range_update_directives

    _, current, sensor = generate_bad_pinchoff_data()
    pf.data["dc_current"].values = pf._normalize_data(current, "dc_current")
    pf.data["dc_sensor"].values = pf._normalize_data(sensor, "dc_sensor")
    pf.prepare_filtered_data()
    pf.find_fit()

    assert "x more negative" in pf.range_update_directives
    assert "x more positive" in pf.range_update_directives


def test_pinchofffit_features_property(nt_dataset_pinchoff, tmp_path):
    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))
    assert not pf._features
    feat_dict = pf.features
    assert feat_dict

    ds = load_by_id(1)
    nt_metadata = json.loads(ds.get_metadata(nt.meta_tag))
    assert nt_metadata["features"] == feat_dict


def test_pinchofffit_fit_result(nt_dataset_pinchoff, tmp_path):
    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))
    fit_result = pf.features

    assert math.isclose(
        fit_result["dc_current"]["amplitude"], 0.5002276487208445, rel_tol=rtol
    )
    assert math.isclose(
        fit_result["dc_current"]["offset"], -22.323634339589656, rel_tol=rtol
    )
    assert math.isclose(
        fit_result["dc_current"]["slope"], 44.645897814266306, rel_tol=rtol
    )
    assert math.isclose(
        fit_result["dc_current"]["residuals"], 0.00325722543290063756, rel_tol=rtol
    )
    val = -0.0504201680672269
    assert math.isclose(
        fit_result["dc_current"]["transition_voltage"], val, rel_tol=rtol
    )
    val = 0.4123686284590007
    assert math.isclose(
        fit_result["dc_current"]["transition_signal"], val, rel_tol=rtol
    )
    assert math.isclose(
        fit_result["dc_current"]["high_signal"], 0.9385653092441386, rel_tol=rtol
    )

    assert math.isclose(
        fit_result["dc_sensor"]["amplitude"], 0.4666022945477287, rel_tol=rtol
    )
    assert math.isclose(
        fit_result["dc_sensor"]["offset"], -14.139712322918397, rel_tol=rtol
    )
    assert math.isclose(
        fit_result["dc_sensor"]["slope"], 28.458715102382286, rel_tol=rtol
    )
    val = 0.11453839742893486
    assert math.isclose(fit_result["dc_sensor"]["residuals"], val, rel_tol=rtol)
    val = -0.0495798319327731
    assert math.isclose(
        fit_result["dc_sensor"]["transition_voltage"], val, rel_tol=rtol
    )
    val = 0.5620627106146739
    assert math.isclose(fit_result["dc_sensor"]["transition_signal"], val, rel_tol=rtol)
    val = 0.8400070918869458
    assert math.isclose(fit_result["dc_sensor"]["high_signal"], val, rel_tol=rtol)


def test_pinchofffit_transition_interval_fitting(nt_dataset_pinchoff, tmp_path):
    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))
    assert not pf._low_signal
    assert not pf._high_signal

    pf.get_transition_from_fit = True
    pf.compute_transition_interval()

    assert math.isclose(pf._high_signal["dc_current"], 0.9328689304577482, rel_tol=rtol)
    assert math.isclose(pf._high_signal["dc_sensor"], 0.8505814391091593, rel_tol=rtol)

    assert math.isclose(pf._low_signal["dc_current"], 0.06741378644796504, rel_tol=rtol)
    assert math.isclose(pf._low_signal["dc_sensor"], 0.07404054174629943, rel_tol=rtol)

    pf.get_transition_from_fit = False
    pf.compute_transition_interval()

    assert math.isclose(pf._high_signal["dc_current"], 0.9385653092441386, rel_tol=rtol)
    assert math.isclose(pf._high_signal["dc_sensor"], 0.8400070918869458, rel_tol=rtol)

    assert math.isclose(
        pf._low_signal["dc_current"], 0.061434690755860326, rel_tol=rtol
    )
    assert math.isclose(pf._low_signal["dc_sensor"], 0.15260464585587372, rel_tol=rtol)


def test_pinchofffit_transition_voltage_fitting(nt_dataset_pinchoff, tmp_path):
    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))
    assert not pf._transition_signal
    assert not pf._transition_signal_index

    pf.get_transition_from_fit = True
    pf.compute_transition_voltage()

    assert math.isclose(
        pf._transition_signal["dc_current"], 0.5926477267201613, rel_tol=rtol
    )
    assert math.isclose(
        pf._transition_signal["dc_sensor"], 0.45264142712057776, rel_tol=rtol
    )

    assert pf._transition_signal_index["dc_current"] == 60
    assert pf._transition_signal_index["dc_sensor"] == 59

    pf.get_transition_from_fit = False
    pf.compute_transition_voltage()

    assert math.isclose(
        pf._transition_signal["dc_current"], 0.4123686284590007, rel_tol=rtol
    )
    assert math.isclose(
        pf._transition_signal["dc_sensor"], 0.5620627106146739, rel_tol=rtol
    )

    assert pf._transition_signal_index["dc_current"] == 59
    assert pf._transition_signal_index["dc_sensor"] == 60


def test_pinchofffit_initial_guess(nt_dataset_pinchoff, tmp_path):
    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))

    bounds, initial_guess = pf.compute_initial_guess(readout_method="dc_current")
    assert math.isclose(bounds[0][0], 0, rel_tol=rtol)
    assert math.isclose(bounds[0][1], 0, rel_tol=rtol)
    assert math.isclose(bounds[0][2], -np.inf, rel_tol=rtol)

    assert math.isclose(bounds[1][0], 1.0, rel_tol=rtol)
    assert math.isclose(bounds[1][1], np.inf, rel_tol=rtol)
    assert math.isclose(bounds[1][2], np.inf, rel_tol=rtol)

    assert math.isclose(initial_guess[0], 1.0, rel_tol=rtol)
    assert math.isclose(initial_guess[1], 1.0, rel_tol=rtol)
    assert math.isclose(initial_guess[2], 0.5, rel_tol=rtol)

    bounds, initial_guess = pf.compute_initial_guess(readout_method="dc_sensor")

    assert math.isclose(bounds[0][0], 0, rel_tol=rtol)
    assert math.isclose(bounds[0][1], 0, rel_tol=rtol)
    assert math.isclose(bounds[0][2], -np.inf, rel_tol=rtol)

    assert math.isclose(bounds[1][0], 0.9957971951822472, rel_tol=rtol)
    assert math.isclose(bounds[1][1], np.inf, rel_tol=rtol)
    assert math.isclose(bounds[1][2], np.inf, rel_tol=rtol)

    assert math.isclose(initial_guess[0], 0.9957971951822472, rel_tol=rtol)
    assert math.isclose(initial_guess[1], 1.0042205429359374, rel_tol=rtol)
    assert math.isclose(initial_guess[2], 0.5, rel_tol=rtol)


def test_pinchoff_fit_fct(nt_dataset_pinchoff, tmp_path):
    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))
    v = np.linspace(-0.5, 0.5, 100)

    with pytest.raises(IndexError):
        _ = pf.fit_fct(v, [1])

    params = [1, 15, 2]
    pf_fit = pf.fit_fct(v, params)

    fit = params[0] * (1 + np.tanh(params[1] * v + params[2]))
    assert np.allclose(fit, pf_fit, rtol=rtol)


def test_pinchoff_fit_plot(nt_dataset_pinchoff, tmp_path):
    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))

    ax, _ = pf.plot_fit(save_figures=True, file_location=str(tmp_path), filename="test")

    assert os.path.exists(os.path.join(str(tmp_path), "test.png"))
    assert len(ax) == len(pf.readout_methods)


def test_pinchoff_features_plot(nt_dataset_pinchoff, tmp_path):
    pf = PinchoffFit(1, "temp.db", db_folder=str(tmp_path))

    ax, _ = pf.plot_features(
        save_figures=True, file_location=str(tmp_path), filename="test"
    )

    assert os.path.exists(os.path.join(str(tmp_path), "test.png"))
    assert len(ax) == len(pf.readout_methods)
