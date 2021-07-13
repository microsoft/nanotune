# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import pytest
from nanotune.device_tuner.tuningresult import TuningResult, MeasurementHistory
from nanotune.device_tuner.tuner import TuningHistory


def test_tuning_history():
    ts = TuningHistory()
    assert ts.results == {}

    meas_res = MeasurementHistory('test_device')
    ts.update('test_device', meas_res)
    assert 'test_device' in ts.results.keys()

    meas_res.add_result(
        TuningResult('gate_characterization', True)
    )
    ts.update('test_device', meas_res)
    assert len(ts.results['test_device'].tuningresults) == 1
    assert ts.results['test_device'].tuningresults['gate_characterization'].success

    ts.update('test_device', TuningResult('characterization', True))

    assert len(ts.results['test_device'].tuningresults) == 2
    assert ts.results['test_device'].tuningresults['characterization'].success
