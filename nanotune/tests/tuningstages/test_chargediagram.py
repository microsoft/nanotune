import pytest

import numpy as np
import nanotune as nt
from nanotune.tuningstages.chargediagram import ChargeDiagram
from nanotune.tests.mock_classifier import MockClassifer
atol = 1e-05


def test_chargediagram_run_stage(chargediagram_settings, experiment):
    chdiag = ChargeDiagram(
        **chargediagram_settings,  # readout_s., setpoint_s, data_s.
        classifiers={'singledot': MockClassifer('singledot'),
                     'doubledot': MockClassifer('doubledot'),
                     'dotregime': MockClassifer('dotregime'),
                    },
        update_settings=False,
    )
    (success,
     termination_reasons,
     tuning_result) = chdiag.run_stage(plot_measurements=False)

    assert success
    assert not termination_reasons
    features = tuning_result['features']

    dc_current_features = features['dc_current']['triple_points']
    assert np.isclose(
        dc_current_features[0][0][0], 0.03511153170221509, atol=atol
        )
    assert np.isclose(
        dc_current_features[1][1][0], 0.020408163265305992, atol=atol
        )

    assert features['dc_sensor']['triple_points']
