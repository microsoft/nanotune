# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging
import pytest
from nanotune.device_tuner.dottuner import DeviceState
from nanotune.device.device_layout import DoubleDotLayout
from nanotune.tests.functional_tests.sim_tuner import SimDotTuner
logger = logging.getLogger(__name__)


def test_double_dot_tuning_good_example(
        tuner_default_input, sim_device, sim_scenario_dottuning
    ):
    device = sim_device
    self = SimDotTuner(
        **tuner_default_input, sim_scenario=sim_scenario_dottuning,
    )
    device_layout = DoubleDotLayout
    target_state = DeviceState.doubledot
    max_iter = 10
    take_high_res = False
    continue_tuning = False

    self.set_central_and_outer_barriers(
        device, device_layout, target_state
    )
    assert device.central_barrier.voltage() == -1.42947649216405
    assert device.left_barrier.voltage() == -0.5141713904634877
    assert device.right_barrier.voltage() == -0.5478492830943646

    done = False
    n_iter = 0
    success = False
    while not done and n_iter <= max_iter:
        n_iter += 1
        self.set_valid_plunger_ranges(device, device_layout)
        assert device.current_valid_ranges()[2] == [-0.331110370123374, -0.0710236745581861]
        assert device.current_valid_ranges()[4] == [-0.312104034678226, -0.0130043347782594]

        tuningresult = self.get_charge_diagram(
            device,
            [device.gates[gid] for gid in device_layout.plungers()],
            iterate=True,
        )
        if tuningresult.success and take_high_res:
            self.take_high_resolution_diagram(
                device,
                device_layout.plungers(),
                take_segments=True,
            )
        done = tuningresult.success
        success = tuningresult.success
        if continue_tuning:
            done = False
            logger.warning("Continue tuning regardless of the outcome.")

        if not done:
            self.update_gate_configuration(
                device, tuningresult, target_state,
            )

    if n_iter >= max_iter:
        logger.info(
            f"Tuning {device.name}: Max number of iterations" \
            f"reached.")

    assert success
    assert n_iter == 1
    self.close()
