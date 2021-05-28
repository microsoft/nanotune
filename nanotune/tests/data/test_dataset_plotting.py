import os

import matplotlib.pyplot as plt
import pytest

import nanotune as nt
from nanotune.data.plotting import plot_dataset


def test_2d_dataset_plotting(nt_dataset_doubledot, tmp_path):
    _, colorbars = plot_dataset(
        1,
        "temp.db",
        db_folder=str(tmp_path),
        save_figures=False,
        file_location=os.path.join(str(tmp_path), "figs"),
    )
    _, colorbars = plot_dataset(
        1,
        "temp.db",
        db_folder=str(tmp_path),
        save_figures=True,
        file_location=str(tmp_path),
    )
    _, colorbars = plot_dataset(
        1,
        "temp.db",
        db_folder=str(tmp_path),
        save_figures=True,
        file_location=os.path.join(str(tmp_path), "figs"),
        filename="testfig",
        plot_filtered_data=True,
    )
    assert colorbars[0].vmax <= 1.05
    assert colorbars[0].vmin >= -0.05


def test_1d_dataset_plotting(nt_dataset_pinchoff, tmp_path):
    ax, _ = plot_dataset(
        1,
        "temp.db",
        db_folder=str(tmp_path),
        save_figures=False,
        file_location=os.path.join(str(tmp_path), "figs"),
        filename="testfig",
    )
    sig_limits = ax[0][0].get_ylim()
    assert sig_limits[1] <= 1.05
    assert sig_limits[0] >= -0.05


def test_dataset_plot_saving(nt_dataset_doubledot, tmp_path):
    ds = nt.Dataset(1, "temp.db", db_folder=str(tmp_path))
    _ = plot_dataset(
        1,
        "temp.db",
        db_folder=str(tmp_path),
        save_figures=True,
        file_location=os.path.join(str(tmp_path), "figs"),
    )
    os.path.exists(os.path.join(str(tmp_path), "dataset_" + str(ds.guid) + ".png"))
    os.path.exists(
        os.path.join(str(tmp_path), "dataset_" + str(ds.guid) + ".eps"),
    )
    _ = plot_dataset(
        1,
        "temp.db",
        db_folder=str(tmp_path),
        save_figures=True,
        file_location=os.path.join(str(tmp_path), "figs"),
        filename="testfig",
    )
    os.path.exists(os.path.join(str(tmp_path), "testfig.png"))
    os.path.exists(os.path.join(str(tmp_path), "dataset.eps"))
