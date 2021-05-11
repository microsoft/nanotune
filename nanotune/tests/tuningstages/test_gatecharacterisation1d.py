import numpy as np
import nanotune as nt
from nanotune.tuningstages.gatecharacterization1d import GateCharacterization1D
from nanotune.tests.mock_classifier import MockClassifer

atol = 1e-05


def test_gatecharacterizaton1D_run(gatecharacterization1D_settings, experiment):
    pinchoff = GateCharacterization1D(
        classifier=MockClassifer("pinchoff"),
        **gatecharacterization1D_settings,  # readout_s., setpoint_s., data_s.
    )

    tuning_result = pinchoff.run_stage(plot_result=False)
    assert tuning_result.success
    assert not tuning_result.termination_reasons
    assert tuning_result.ml_result
