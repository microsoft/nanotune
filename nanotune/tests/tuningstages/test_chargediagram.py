import numpy as np
import pytest

import qcodes as qc
import nanotune as nt
from nanotune.tests.mock_classifier import MockClassifer
from nanotune.tuningstages.chargediagram import ChargeDiagram

atol = 1e-03


def test_chargediagram_run_stage(chargediagram_settings):

    chdiag = ChargeDiagram(
        **chargediagram_settings,  # readout_s., setpoint_s, data_s.
        classifiers={
            "singledot": MockClassifer("singledot"),
            "doubledot": MockClassifer("doubledot"),
            "dotregime": MockClassifer("dotregime"),
        },
    )
    tuning_result = chdiag.run_stage(plot_result=False)
    assert tuning_result.success
    assert not tuning_result.termination_reasons
    features = tuning_result.ml_result["features"]
    print(features)

    dc_current_features = features["dc_current"]["triple_points"]
    assert features["dc_current"]["triple_points"]
    assert features["dc_sensor"]["triple_points"]
    assert np.isclose(dc_current_features[0][0][0], 0.03583013287130833, atol=atol)
