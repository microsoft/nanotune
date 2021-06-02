# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import pytest
from nanotune.tuningstages.gatecharacterization_tasks import *

def test_get_new_gatecharacterization_range():

    current_valid_ranges = [(-1, -0.5)]
    safety_voltage_ranges = [(-3, 0)]
    range_update_directives = ['x more negative']

    new_range = get_new_gatecharacterization_range(
        current_valid_ranges, safety_voltage_ranges, ['x more negative']
    )
    assert new_range == [(-3, -0.5)]

    new_range = get_new_gatecharacterization_range(
        current_valid_ranges, safety_voltage_ranges, ['x more positive']
    )
    assert new_range == [(-1, 0)]

    with pytest.raises(KeyError):
        new_range = get_new_gatecharacterization_range(
            current_valid_ranges, safety_voltage_ranges, ['get ice cream']
        )

    with pytest.raises(AssertionError):
        new_range = get_new_gatecharacterization_range(
            (-1, -0.5), safety_voltage_ranges, ['x more negative']
        )
    with pytest.raises(AssertionError):
        new_range = get_new_gatecharacterization_range(
            current_valid_ranges, (-3, 0), ['x more negative']
        )

def test_get_range_directives_gatecharacterization():
    current_valid_ranges = [(-1, -0.5)]
    safety_voltage_ranges = [(-3, 0)]

    directives, issues = get_range_directives_gatecharacterization(
        ["x more negative", "x more positive"],
        current_valid_ranges,
        safety_voltage_ranges,
    )
    assert directives == ["x more negative", "x more positive"]
    assert issues == []

    with pytest.raises(KeyError):
        directives, issues = get_range_directives_gatecharacterization(
            ["get chocolate"],
            current_valid_ranges,
            safety_voltage_ranges,
        )

    directives, issues = get_range_directives_gatecharacterization(
        ["x more positive", "x more negative"],
        [(-2.99, -0.001)],
        safety_voltage_ranges,
    )
    assert directives == []
    assert "positive safety voltage reached" in issues
    assert "negative safety voltage reached" in issues

    directives, issues = get_range_directives_gatecharacterization(
        ["x more positive", "x more negative"],
        [(-2.99, -0.22)],
        safety_voltage_ranges,
        dV_stop=0.2,
    )
    assert directives == ["x more positive"]
    assert issues == ["negative safety voltage reached"]

def test_finish_early_pinched_off():

    last_measurement_strength = 0.015
    normalization_constant = (0., 1.)
    recent_measurement_strengths = [0.011, 0.01, 0.011, 0.0105]
    voltage_precision = 0.1
    noise_level = np.mean([0.011, 0.0105, 0.015])*1.1

    finish, new_recent_output = finish_early_pinched_off(
        last_measurement_strength,
        normalization_constant,
        recent_measurement_strengths,
        voltage_precision,
        noise_level,
        voltage_interval_to_track=0.3,
    )
    assert finish
    assert new_recent_output == [0.011, 0.0105, 0.015]

    noise_level = np.mean([0.01, 0.011, 0.0105, 0.015])*0.9
    finish, new_recent_output = finish_early_pinched_off(
        last_measurement_strength,
        normalization_constant,
        recent_measurement_strengths,
        voltage_precision,
        noise_level,
        voltage_interval_to_track=0.35,
    )
    assert not finish
    assert new_recent_output == [0.01, 0.011, 0.0105, 0.015]

    finish, new_recent_output = finish_early_pinched_off(
        last_measurement_strength,
        normalization_constant,
        [],
        voltage_precision,
        noise_level,
        voltage_interval_to_track=0.35,
    )
    assert not finish
    assert new_recent_output == [0.015]
