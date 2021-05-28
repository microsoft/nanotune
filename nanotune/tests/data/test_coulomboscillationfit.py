import os
import json
import math
import pytest
import numpy as np
from qcodes.dataset.experiment_container import load_by_id
import nanotune as nt
from nanotune.fit.coulomboscillationfit import CoulombOscillationFit

rtol = 1e-05


def test_coulomboscillationfit_init(nt_dataset_coulomboscillation, tmp_path):
    attributes = [
        "relative_height_threshold",
        "sigma_dV",
        "peak_indx",
        "peak_distances",
    ]
    co = CoulombOscillationFit(1, "temp.db", db_folder=str(tmp_path))

    for attr in attributes:
        assert hasattr(co, attr)
        getattr(co, attr)


def test_coulomboscillationfit_range_update_directives(
    nt_dataset_coulomboscillation, tmp_path
):
    co = CoulombOscillationFit(1, "temp.db", db_folder=str(tmp_path))
    with pytest.raises(NotImplementedError):
        co.range_update_directives


def test_coulomboscillationfit_features(nt_dataset_coulomboscillation, tmp_path):
    co = CoulombOscillationFit(1, "temp.db", db_folder=str(tmp_path))
    assert not co._features
    feat_dict = co.features
    assert feat_dict

    ds = load_by_id(1)
    nt_metadata = json.loads(ds.get_metadata(nt.meta_tag))
    assert nt_metadata["features"] == feat_dict


def test_coulomboscillationfit_fit_result(nt_dataset_coulomboscillation, tmp_path):
    co = CoulombOscillationFit(1, "temp.db", db_folder=str(tmp_path))
    fit_result = co.features

    target_peak_loc_dc_current = [
        -0.899159663865546,
        -0.798319327731092,
        -0.600840336134454,
    ]
    target_peak_loc_dc_sensor = [
        -0.949579831932773,
        -0.848739495798319,
        -0.651260504201681,
    ]

    assert [24, 48, 95] == sorted(fit_result["dc_current"]["peak_indx"])
    assert [12, 36, 83] == sorted(fit_result["dc_sensor"]["peak_indx"])
    for ii in range(3):
        assert math.isclose(
            fit_result["dc_current"]["peak_locations"][ii],
            target_peak_loc_dc_current[ii],
            rel_tol=rtol,
        )
        assert math.isclose(
            fit_result["dc_sensor"]["peak_locations"][ii],
            target_peak_loc_dc_sensor[ii],
            rel_tol=rtol,
        )

    target_peak_dist_dc_current = [0.10084033613445398, 0.19747899159663806]
    target_peak_dist_dc_sensor = [0.10084033613445398, 0.19747899159663795]
    for ii in range(2):
        assert math.isclose(
            fit_result["dc_current"]["peak_distances"][ii],
            target_peak_dist_dc_current[ii],
            rel_tol=rtol,
        )
        assert math.isclose(
            fit_result["dc_sensor"]["peak_distances"][ii],
            target_peak_dist_dc_sensor[ii],
            rel_tol=rtol,
        )


# def test_coulomboscillationfit_peak_distances(nt_dataset_coulomboscillation,
#                                               tmp_path):
#     co = CoulombOscillationFit(1, 'temp.db', db_folder=str(tmp_path))

#     peak_distances = co.calculate_peak_distances()


def test_coulomboscillationfit_plot(nt_dataset_coulomboscillation, tmp_path):
    co = CoulombOscillationFit(1, "temp.db", db_folder=str(tmp_path))

    ax, _ = co.plot_fit(save_figures=True, file_location=str(tmp_path), filename="test")

    assert os.path.exists(os.path.join(str(tmp_path), "test.png"))
    assert len(ax) == len(co.data)
