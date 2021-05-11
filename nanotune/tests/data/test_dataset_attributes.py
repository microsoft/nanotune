import pytest
import json
import numpy as np
from qcodes.dataset.experiment_container import load_by_id
import nanotune as nt
from nanotune.data.dataset import Dataset


def test_dataset_attributes_after_init(nt_dataset_doubledot, tmp_path):
    """
    Ensure that all attributes are populated after initialisation.
    Also tests property getters
    """
    attributes = [
        "db_name",
        "db_folder",
        "qc_run_id",
        "snapshot",
        "normalization_constants",
        "device_name",
        "readout_methods",
        "quality",
        "label",
        "raw_data",
        "data",
        "power_spectrum",
        "filtered_data",
        "dimensions",
        "exp_id",
    ]

    ds = Dataset(1, db_name="temp.db", db_folder=str(tmp_path))

    for attr in attributes:
        assert hasattr(ds, attr)
        getattr(ds, attr)

    assert ds.device_name == "test_device"
    assert ds._normalization_constants["dc_current"] == [0, 2]
    assert ds._normalization_constants["rf"] == [0, 1]
    assert ds._normalization_constants["dc_sensor"] == [-0.32, 3]
    assert ds.readout_methods == {"dc_current": "current", "dc_sensor": "sensor"}
    assert ds.qc_run_id == 1
    assert ds.db_name == "temp.db"
    assert ds.quality == 1
    assert set(ds.label) == {"doubledot"}
    # Check if signal has been normalized
    # ds.signal_raw: not normalized
    # ds.signal: normalized and nans removed
    # assert np.max(ds.signal[0]) <= 1


def test_dataset_defaults_for_missing_metadata(
    nt_dataset_doubledot_partial_metadata, tmp_path
):
    ds = Dataset(1, db_name="temp.db", db_folder=str(tmp_path))

    assert ds.normalization_constants["dc_current"] == [0, 1.4]
    assert ds.normalization_constants["rf"] == [0, 1]
    assert ds.normalization_constants["dc_sensor"] == [0, 1]
    assert ds.readout_methods == {"dc_current": "current", "dc_sensor": "sensor"}
    assert ds.device_name == "noname_device"
    assert ds.quality is None
    assert ds.label == ["doubledot"]


def test_dataset_property_getters(nt_dataset_pinchoff, tmp_path):
    ds = Dataset(1, db_name="temp.db", db_folder=str(tmp_path))

    assert ds._normalization_constants == ds.normalization_constants
    assert ds.features == {
        "dc_current": {
            "amplitude": 0.6,
            "slope": 1000,
            "low_signal": 0,
            "high_signal": 1,
            "residuals": 0.5,
            "offset": 50,
            "transition_signal": 0.5,
            "low_voltage": -0.06,
            "high_voltage": -0.03,
            "transition_voltage": -0.05,
        },
        "dc_sensor": {
            "amplitude": 0.5,
            "slope": 800,
            "low_signal": 0,
            "high_signal": 1,
            "residuals": 0.5,
            "offset": 50,
            "transition_signal": 0.5,
            "low_voltage": -0.06,
            "high_voltage": -0.03,
            "transition_voltage": -0.05,
        },
    }
    assert ds._snapshot == ds.snapshot
    qc_ds = load_by_id(1)
    assert ds.snapshot == json.loads(qc_ds.get_metadata("snapshot"))
    nt_metadata = json.loads(qc_ds.get_metadata(nt.meta_tag))
    assert ds.nt_metadata == nt_metadata

    with pytest.raises(AttributeError):
        ds.snapshot = {}

    with pytest.raises(AttributeError):
        ds.normalization_constants = {}

    with pytest.raises(AttributeError):
        ds.features = {}

    qc_ds = load_by_id(1)
    assert ds.features == nt_metadata["features"]
