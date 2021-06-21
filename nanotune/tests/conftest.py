import gc
import os

import shutil
import pytest
import qcodes as qc
from qcodes import new_experiment
from qcodes.tests.instrument_mocks import DummyInstrument

import nanotune as nt
from nanotune.device.device import Device
from nanotune.device.device import DeviceChannel
from nanotune.tests.data_generator_methods import (
    DotCurrent, DotSensor, PinchoffCurrent, PinchoffSensor,
    generate_coloumboscillation_metadata,
    generate_coulomboscillations,
    generate_doubledot_data, generate_doubledot_metadata,
    generate_pinchoff_data, generate_pinchoff_metadata,
    populate_db_doubledots, populate_db_pinchoffs)

from nanotune.drivers.mock_dac import MockDAC, MockDACChannel
from nanotune.drivers.mock_readout_instruments import MockLockin, MockRF

from .data_savers import save_1Ddata_with_qcodes, save_2Ddata_with_qcodes

ideal_run_labels = {
    0: "pinchoff",
    1: "pinchoff",
    2: "pinchoff",
    3: "pinchoff",
    4: "pinchoff",
    5: "doubledot",
}


@pytest.fixture(scope="function")
def empty_temp_db(tmp_path):
    global n_experiments
    n_experiments = 0
    try:
        nt.new_database("temp.db", tmp_path)
        yield
    finally:
        gc.collect()


@pytest.fixture(scope="function")
def second_empty_temp_db(tmp_path):
    global n_experiments
    n_experiments = 0
    try:
        nt.new_database("temp2.db", tmp_path)
        yield
    finally:
        gc.collect()


@pytest.fixture(scope="function")
def empty_db_different_folder(tmp_path):
    global n_experiments
    n_experiments = 0
    try:
        path = os.path.join(str(tmp_path), "test")
        os.mkdir(path)
        nt.new_database("temp2.db", path)
        yield
    finally:
        gc.collect()


@pytest.fixture(scope="function")
def experiment(empty_temp_db):
    e = new_experiment("test-experiment", sample_name="test_sample")
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope="function")
def second_experiment_second_db(second_empty_temp_db):
    e = new_experiment("test-experiment2", sample_name="test_sample")
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope="function")
def experiment_different_db_folder(empty_db_different_folder):
    e = new_experiment("test-experiment-duh", sample_name="test_sample")
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(name="dummy_dmm", scope="function")
def _make_dummy_inst():
    inst = DummyInstrument("dummy_dmm")
    inst.dac1.get = lambda: 0.001
    inst.dac2.get = lambda: 1.4
    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture(scope="function", params=["numeric"])
def qc_dataset_pinchoff(experiment, request):
    """"""
    datasaver = save_1Ddata_with_qcodes(generate_pinchoff_data, None)

    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function", params=["numeric"])
def nt_dataset_pinchoff(experiment, request):
    """"""
    datasaver = save_1Ddata_with_qcodes(
        generate_pinchoff_data, generate_pinchoff_metadata
    )
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function")
def db_real_pinchoff(tmp_path):

    nt_path = os.path.dirname(os.path.dirname(os.path.abspath(nt.__file__)))
    path_device_characterization = os.path.join(
        nt_path, 'data', 'tuning', 'device_characterization.db')

    try:
        shutil.copyfile(
            path_device_characterization,
            os.path.join(tmp_path, "pinchoff_data.db"))
        # nt.new_database("pinchoff_data.db", tmp_path)
        # extract_runs_into_db(
        #     path_device_characterization,
        #     os.path.join(tmp_path, 'pinchoff_data.db'),
        #     *list(range(1, 16)),
        # )
        yield
    finally:
        gc.collect()


@pytest.fixture(scope="function", params=["numeric"])
def qc_dataset_doubledot(experiment, request):
    """"""
    datasaver = save_2Ddata_with_qcodes(generate_doubledot_data, None)

    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function")
def nt_dataset_doubledot(experiment, tmp_path):
    datasaver = save_2Ddata_with_qcodes(
        generate_doubledot_data, generate_doubledot_metadata
    )

    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function")
def experiment_pinchoffs(empty_temp_db):
    e = new_experiment("test_experiment", sample_name="test_sample")
    populate_db_pinchoffs()
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope="function")
def experiment_doubledots(empty_temp_db):
    e = new_experiment("test_experiment", sample_name="test_sample")
    populate_db_doubledots()
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope="function", params=["numeric"])
def nt_dataset_coulomboscillation(experiment, request):
    """"""
    datasaver = save_1Ddata_with_qcodes(
        generate_coulomboscillations, generate_coloumboscillation_metadata
    )
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function")
def experiment_ideal_run(empty_temp_db):
    e = new_experiment("test_tuning_run", sample_name="test_sample")
    for run_id in range(len(ideal_run_labels)):
        label = ideal_run_labels[run_id]
        if label == "pinchoff":
            _ = save_1Ddata_with_qcodes(
                generate_pinchoff_data,
                generate_pinchoff_metadata,
            )
        elif label == "doubledot":
            _ = save_2Ddata_with_qcodes(
                generate_doubledot_data,
                generate_doubledot_metadata,
            )
        else:
            raise NotImplementedError
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope="function")
def pinchoff_dmm(dummy_dmm):
    dummy_dmm.add_parameter(
        "po_current",
        parameter_class=PinchoffCurrent,
        initial_value=0,
        label="pinchoff dc current",
        unit="A",
        get_cmd=None,
        set_cmd=None,
    )

    dummy_dmm.add_parameter(
        "po_sensor",
        parameter_class=PinchoffSensor,
        initial_value=0,
        label="pinchoff sensor",
        unit="A",
        get_cmd=None,
        set_cmd=None,
    )
    try:
        yield dummy_dmm
    finally:
        dummy_dmm.close()


@pytest.fixture(scope="function")
def dot_dmm(dummy_dmm):
    dummy_dmm.add_parameter(
        "dot_current",
        parameter_class=DotCurrent,
        label="dot current",
        unit="A",
        get_cmd=None,
        set_cmd=None,
    )

    dummy_dmm.add_parameter(
        "dot_sensor",
        parameter_class=DotSensor,
        label="dot sensor",
        unit="A",
        get_cmd=None,
        set_cmd=None,
    )
    try:
        yield dummy_dmm
    finally:
        dummy_dmm.close()


@pytest.fixture(scope="session")
def dac():
    return MockDAC('dac', MockDACChannel)


@pytest.fixture(scope="session")
def lockin():
    _lockin = MockLockin(
        name='lockin'
    )
    return _lockin


@pytest.fixture(scope="session")
def rf():
    return MockRF('rf')


@pytest.fixture(scope="session")
def station(dac, lockin, rf):
    _station = qc.Station()
    _station.add_component(dac)
    _station.add_component(lockin)
    _station.add_component(rf)
    return _station


@pytest.fixture(scope="function")
def gate_1(station):
    gate = DeviceChannel(
        station.dac,
        'test_gate',
        station.dac.ch01,
        gate_id=1,
        label="test_label",
        delay=0,
        max_voltage_step=0.01,
        ramp_rate=0.2,
    )
    yield gate


@pytest.fixture(scope="function")
def gate_2(station):
    gate = DeviceChannel(
        station.dac,
        'test_gate_1',
        station.dac.ch02,
        gate_id=1,
        label="test_label_2",
        delay=0,
        max_voltage_step=0.01,
        ramp_rate=0.2,
    )
    yield gate


@pytest.fixture(scope="function")
def gatecharacterization1D_settings(pinchoff_dmm, gate_1, tmp_path):
    gate_1.current_valid_range([-0.1, 0])
    pinchoff_dmm.po_current.gate = gate_1
    pinchoff_dmm.po_sensor.gate = gate_1

    readout_methods = {
        "transport": pinchoff_dmm.po_current,
        "sensing": pinchoff_dmm.po_sensor,
    }
    setpoint_settings = {
        "voltage_precision": 0.001,
        "parameters_to_sweep": [gate_1.dc_voltage],
        "current_valid_ranges": [gate_1.current_valid_range()],
        "safety_voltage_ranges": [(-3, 0)],
    }
    data_settings = {
        "db_name": "temp.db",
        "normalization_constants": {
            "transport": [0, 1.2],
            "sensing": [-0.13, 1.1],
            "rf": [0, 1],
        },
        "db_folder": tmp_path,
    }
    tuningstage_settings = {
        "readout_methods": readout_methods,
        "setpoint_settings": setpoint_settings,
        "data_settings": data_settings,
    }
    yield tuningstage_settings


@pytest.fixture(scope="function")
def chargediagram_settings(dot_dmm, tmp_path, gate_1, gate_2):
    gate_1.current_valid_range([-0.2, -0.1])
    gate_2.current_valid_range([-0.3, -0.2])

    dot_dmm.dot_current.gate_x = gate_1
    dot_dmm.dot_current.gate_y = gate_2

    dot_dmm.dot_sensor.gate_x = gate_1
    dot_dmm.dot_sensor.gate_y = gate_2

    readout_methods = {
        "transport": dot_dmm.dot_current,
        "sensing": dot_dmm.dot_sensor,
    }
    setpoint_settings = {
        "voltage_precision": 0.001,
        "parameters_to_sweep": [gate_1.dc_voltage, gate_2.dc_voltage],
        "current_valid_ranges": [
            gate_1.current_valid_range(),
            gate_2.current_valid_range(),
        ],
        "safety_voltage_ranges": [(-3, 0), (-3, 0)],
    }
    data_settings = {
        "db_name": "temp.db",
        "segment_db_name": "temp_dots_seg.db",
        "segment_db_folder": tmp_path,
        "normalization_constants": {
            "transport": [0, 2],
            "sensing": [-0.32, 3],
            "rf": [0, 1],
        },
        "db_folder": tmp_path,
        "segment_size": 0.05,
    }

    tuningstage_settings = {
        "readout_methods": readout_methods,
        "setpoint_settings": setpoint_settings,
        "data_settings": data_settings,
    }
    yield tuningstage_settings


@pytest.fixture(scope="function")
def dot_readout_methods(dot_dmm):
    readout_methods = {
        "transport": dot_dmm.dot_current,
        "sensing": dot_dmm.dot_sensor,
    }
    yield readout_methods


@pytest.fixture(scope="function")
def device_pinchoff(pinchoff_dmm, device_gate_inputs):
    readout_methods = {
        "transport": pinchoff_dmm.po_current,
        "sensing": pinchoff_dmm.po_sensor,
    }

    device = nt.Device(
        name="test_doubledot",
        device_type="doubledot_2D",
        readout_methods=readout_methods,
        **device_gate_inputs,
    )

    pinchoff_dmm.po_current.gate = device.left_barrier
    pinchoff_dmm.po_sensor.gate = device.left_barrier
    try:
        yield device
    finally:
        device.close()
