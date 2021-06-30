# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest
from nanotune.device_tuner.characterizer import Characterizer


def test_characterizer_characterize(
    tuner_default_input,
    sim_device,
):
    tuner = Characterizer(**tuner_default_input)
    res = tuner.characterize(
        sim_device,
        skip_gates=sim_device.gates,
    )
    assert res.device_name == sim_device.name
    assert len(res.to_dict()) == 1

    tuner.close()