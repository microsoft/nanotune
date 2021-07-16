import json

import pytest

from nanotune.device_tuner.tuningresult import MeasurementHistory, TuningResult


def test_tuningresult():
    ts = TuningResult(
        "gatecharacterization",
        True,
        ["aaaaaaaa-0000-0000-0000-000000000000"],
        data_ids=[1],
    )

    assert ts.termination_reasons == []
    assert ts.success
    assert ts.guids == ["aaaaaaaa-0000-0000-0000-000000000000"]
    assert ts.data_ids == [1]
    assert ts.timestamp == ""
    assert ts.ml_result == {}

    ts_dict = ts.to_dict()
    assert isinstance(ts.to_dict(), dict)
    for attr in ["termination_reasons", "success", "data_ids", "timestamp"]:
        assert attr in ts_dict.keys()

    assert isinstance(ts.to_json(), str)
    assert json.dumps(ts_dict) == ts.to_json()


def test_measurement_history_attributes():
    ms = MeasurementHistory("doubledot")

    assert not ms.tuningresults
    assert ms.device_name == "doubledot"

    ts = TuningResult(
        "gatecharacterization", True, ["aaaaaaaa-0000-0000-0000-000000000000"]
    )
    ms.add_result(ts)
    assert ms.tuningresults[ts.stage] == ts
    ms.add_result(ts)
    assert len(ms.tuningresults) == 1
    ts2 = TuningResult(
        "gatecharacterization", True, ["aaaaaaaa-0000-0000-0000-000000000001"]
    )

    ms.add_result(ts2)
    assert len(ms.tuningresults) == 2
    new_key = "gatecharacterization" + "_" + "aaaaaaaa-0000-0000-0000-000000000001"
    assert ms.tuningresults[new_key] == ts2

    ts3 = TuningResult("chargediagram", True, ["aaaaaaaa-0000-0000-0000-000000000002"])
    ms.add_result({"test_diagram": ts3})
    assert ms.tuningresults["test_diagram"] == ts3

    ts4 = TuningResult("chargediagram", True, ["aaaaaaaa-0000-0000-0000-000000000003"])
    ms.add_result(ts4, "another_diagram")
    assert len(ms.tuningresults) == 4
    assert ms.tuningresults["another_diagram"] == ts4

    ts5 = TuningResult("chargediagram", True, ["aaaaaaaa-0000-0000-0000-000000000004"])
    ms.add_result(ts5)
    assert len(ms.tuningresults) == 5
    assert ms.tuningresults["chargediagram"] == ts5


def test_measurement_history_exports():
    ms = MeasurementHistory("doubledot")
    ts = TuningResult(
        "gatecharacterization", True, ["aaaaaaaa-0000-0000-0000-000000000000"]
    )
    ms.add_result(ts, "test")

    ms_dict = ms.to_dict()
    assert isinstance(ms_dict, dict)
    assert ms_dict["device_name"] == "doubledot"
    assert ms_dict["test"] == ts.to_dict()

    assert json.dumps(ms_dict) == ms.to_json()
