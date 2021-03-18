import numpy as np
import nanotune as nt
from nanotune.tuningstages.gatecharacterization1d import GateCharacterization1D
from nanotune.tests.mock_classifier import MockClassifer
atol = 1e-05

# check that setpoints do not exceed max_ramp of each gate

def test_gatecharacterizaton1D_run(gatecharacterization1D_settings, experiment):
    pinchoff = GateCharacterization1D(
        classifier=MockClassifer('pinchoff'),
        **gatecharacterization1D_settings, # readout_s., setpoint_s., data_s.
        update_settings=False,
    )
    (success,
     termination_reasons,
     tuning_result) = pinchoff.run_stage(plot_measurements=False)

    assert success
    assert not termination_reasons
    features = tuning_result['features']
    print(tuning_result)

    # assert np.isclose(features['dc_current']['amplitude'], 0.5002276487208445, atol=atol)