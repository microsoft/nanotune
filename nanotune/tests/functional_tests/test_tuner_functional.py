import pytest
from numpy.testing import assert_almost_equal
from nanotune.tuningstages.settings import Classifiers
from nanotune.tests.functional_tests.sim_tuner import SimDotTuner


def test_measure_initial_ranges_2D(
    tuner_default_input,
    sim_device,
    sim_scenario_init_ranges,
    pinchoff_classifier,
):
    device = sim_device
    tuner = SimDotTuner(
        **tuner_default_input, sim_scenario=sim_scenario_init_ranges,
    )
    tuner.classifiers = Classifiers(pinchoff=pinchoff_classifier)

    ((min_voltage, max_voltage),
     measurement_result) = tuner.measure_initial_ranges_2D(
        sim_device,
        gate_to_set = device.top_barrier,
        gates_to_sweep=[
            device.central_barrier, device.left_barrier, device.right_barrier
        ],
        voltage_step = 0.2,
    )

    assert_almost_equal(min_voltage, -0.6428571428571428)
    assert len(measurement_result.to_dict()) == 14
    assert_almost_equal(max_voltage, -2.918972990997)
    tuner.close()
