import math
import os

import numpy as np
import pytest

import nanotune as nt
from nanotune.fit.dotfit import DotFit
from nanotune.tests.data.conftest import generate_doubledot_data

atol = 1e-05


def test_dotfit_init(nt_dataset_doubledot, tmp_path):

    attributes = [
        "signal_thresholds",
        "segment_size",
        "triple_points",
        "fit_parameters",
        "segmented_data",
    ]

    df = DotFit(1, "temp.db", db_folder=str(tmp_path), signal_thresholds=[0.03, 0.4])

    for attr in attributes:
        assert hasattr(df, attr)
        getattr(df, attr)


def test_dotfit_range_update_directives(nt_dataset_doubledot, tmp_path):
    df = DotFit(
        1,
        "temp.db",
        db_folder=str(tmp_path),
        signal_thresholds=[0.03, 0.4],
    )
    assert not df.range_update_directives

    edge = df.get_edge("left vertical", use_raw_data=False)
    assert np.max(edge) > 0.24
    assert np.min(edge) < 0.06

    edge = df.get_edge("right vertical", use_raw_data=False)
    assert np.max(edge) > 0.2
    assert np.min(edge) < 0.03

    edge = df.get_edge("bottom horizontal", use_raw_data=False)
    assert np.max(edge) > 0.26
    assert np.min(edge) < 0.02

    edge = df.get_edge("top horizontal", use_raw_data=False)
    assert np.max(edge) > 0.25
    assert np.min(edge) < 0.03

    df.signal_thresholds = [0.1, 0.2]
    df._range_update_directives = []
    assert "x more negative" in df.range_update_directives
    assert "y more negative" in df.range_update_directives

    df.signal_thresholds = [0.5, 0.7]
    df._range_update_directives = []
    assert "x more positive" in df.range_update_directives
    assert "y more positive" in df.range_update_directives


def test_dotfit_fit(nt_dataset_doubledot, tmp_path):
    df = DotFit(1, "temp.db", db_folder=str(tmp_path))
    df.find_fit()
    assert df.features


def test_dotfit_segment_data(nt_dataset_doubledot, tmp_path):
    segment_size = 0.05
    df = DotFit(
        1,
        "temp.db",
        db_folder=str(tmp_path),
        segment_size=segment_size,
    )

    df.prepare_segmented_data(use_raw_data=True)

    assert len(df.segmented_data) == 4
    assert len(df.segmented_data[0]) == 2

    xv, yv, ddot, sensor = generate_doubledot_data()
    voltage_x = np.unique(xv)
    voltage_y = np.unique(yv)

    dot_shape_y, dot_shape_x = ddot.shape

    vx_span = abs(voltage_x[0] - voltage_x[-1])
    vx_span = round(vx_span, 8)
    n_x = int(math.floor(vx_span / segment_size))

    vy_span = abs(voltage_y[0] - voltage_y[-1])
    vy_span = round(vy_span, 8)
    n_y = int(math.floor(vy_span / segment_size))

    d_inx = int(dot_shape_x / n_x)
    d_iny = int(dot_shape_y / n_y)

    dot_segment = df.segmented_data[0]["current"].values
    assert np.allclose(dot_segment.T, ddot[:d_iny, :d_inx], rtol=atol)

    sensor_segment = df.segmented_data[0]["sensor"].values
    assert np.allclose(sensor_segment.T, sensor[:d_iny, :d_inx], rtol=atol)

    v_x_segment = df.segmented_data[0]["current"]["v_x"].values
    assert np.allclose(v_x_segment, voltage_x[:d_inx], rtol=atol)

    v_y_segment = df.segmented_data[0]["current"]["v_y"].values
    assert np.allclose(v_y_segment, voltage_y[:d_iny], rtol=atol)


def test_dotfit_save_segmented_data(
    nt_dataset_doubledot, tmp_path, experiment_different_db_folder
):
    segment_size = 0.05
    df = DotFit(1, "temp.db", db_folder=str(tmp_path), segment_size=segment_size)

    db_folder2 = os.path.join(str(tmp_path), "test")
    seg_info = df.save_segmented_data_return_info(
        segment_db_name="temp2.db", segment_db_folder=db_folder2
    )

    db_name, db_folder = nt.get_database()
    assert db_name == "temp.db"
    assert db_folder == str(tmp_path)

    assert sorted(seg_info.keys()) == [1, 2, 3, 4]

    assert seg_info[1]["voltage_ranges"] == [(-0.2, -0.15125), (-0.3, -0.252)]

    assert seg_info[2]["voltage_ranges"] == [(-0.2, -0.15125), (-0.25, -0.2)]

    assert seg_info[3]["voltage_ranges"] == [(-0.15, -0.1), (-0.3, -0.252)]

    assert seg_info[4]["voltage_ranges"] == [(-0.15, -0.1), (-0.25, -0.2)]

    seg_df = nt.Dataset(1, "temp2.db", db_folder2)
    raw_current_new_df = seg_df.raw_data["current"].values

    raw_current_original = df.segmented_data[0]["current"].values

    assert np.allclose(raw_current_new_df, raw_current_original, rtol=atol)


def test_get_triple_point_distances(nt_dataset_doubledot, tmp_path):
    df = DotFit(1, "temp.db", db_folder=str(tmp_path))

    df.fit_parameters = {
        "transport": {
            "noise_level": 0.02,
            "binary_neighborhood": 1,
            "distance_threshold": 0.05,
        },
        "sensing": {
            "noise_level": 0.3,
            "binary_neighborhood": 2,
            "distance_threshold": 0.05,
        },
    }

    relevant_distances = df.get_triple_point_distances()
    print(relevant_distances["sensing"])

    assert len(relevant_distances["transport"]) == 2
    transport_distance = np.asarray(relevant_distances["transport"])
    found = False
    for dist in transport_distance[:, 0, 0]:
        if math.isclose(0.036055512754639925, dist, rel_tol=atol):
            found = True
    assert found

    assert [-0.14, -0.28] in transport_distance[:, 2]
    assert [-0.17, -0.22] in transport_distance[:, 3]
    # fit results of sensor
    sensing_distance = np.asarray(relevant_distances["sensing"])
    assert len(sensing_distance) == 2
    found = False
    for dist in sensing_distance[:, 0, 0]:
        if math.isclose(0.0353774292452123, dist, rel_tol=atol):
            found = True
    assert found

    assert [-0.19, -0.25] in sensing_distance[:, 2]
    assert [-0.12, -0.25] in sensing_distance[:, 3]


def test_dotfit_plot(nt_dataset_doubledot, tmp_path):
    df = DotFit(1, "temp.db", db_folder=str(tmp_path))

    df.fit_parameters = {
        "transport": {
            "noise_level": 0.02,
            "binary_neighborhood": 1,
            "distance_threshold": 0.05,
        },
        "sensing": {
            "noise_level": 0.3,
            "binary_neighborhood": 2,
            "distance_threshold": 0.05,
        },
    }
    df.triple_points = df.get_triple_point_distances()
    _, colorbars = df.plot_fit(
        save_figure=True, file_location=str(tmp_path), filename="test"
    )

    assert os.path.exists(os.path.join(str(tmp_path), "test.png"))
    assert len(colorbars) == len(df.data)
