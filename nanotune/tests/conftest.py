import gc
import os

import shutil
import pytest
import pathlib
import qcodes as qc
from qcodes import new_experiment
from qcodes.tests.instrument_mocks import DummyInstrument

import nanotune as nt
from nanotune.tuningstages.settings import DataSettings, SetpointSettings
from nanotune.device.device import NormalizationConstants, ReadoutMethods
from nanotune.device.device_channel import DeviceChannel
from nanotune.classification.classifier import Classifier
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
PARENT_DIR = pathlib.Path(__file__).parent.absolute()

from nanotune.tests.mock_classifier import MockClassifer
from nanotune.tuningstages.settings import (DataSettings, SetpointSettings,
    Classifiers)
from nanotune.device_tuner.tuner import Tuner
from sim.data_providers import QcodesDataProvider
from sim.qcodes_mocks import MockDoubleQuantumDotInstrument

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
        yield
    finally:
        gc.collect()


@pytest.fixture(scope="function")
def db_dot_tuning(tmp_path):

    nt_path = os.path.dirname(os.path.dirname(os.path.abspath(nt.__file__)))
    path_device_characterization = os.path.join(
        nt_path, 'data', 'tuning', 'dot_tuning_sequences.db')

    try:
        shutil.copyfile(
            path_device_characterization,
            os.path.join(tmp_path, "dot_tuning_data.db"))
        yield
    finally:
        gc.collect()


@pytest.fixture(scope="function", params=["numeric"])
def qc_dataset_doubledot(experiment, request):
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
        label="pinchoff transport",
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


@pytest.fixture(scope="function")
def dac():
    _dac = MockDAC('dac', MockDACChannel)
    try:
        yield _dac
    finally:
        _dac.close()



@pytest.fixture(scope="function")
def lockin():
    _lockin = MockLockin(
        name='lockin'
    )
    try:
        yield _lockin
    finally:
        _lockin.close()


@pytest.fixture(scope="function")
def rf():
    _rf = MockRF('rf')
    try:
        yield _rf
    finally:
        _rf.close()


@pytest.fixture(scope="function")
def station(dac, lockin, rf):
    _station = qc.Station()
    _station.add_component(dac)
    _station.add_component(lockin)
    _station.add_component(rf)
    try:
        yield _station
    finally:
        _station.close_all_registered_instruments()


@pytest.fixture(scope="function")
def gate_1(station):
    gate = DeviceChannel(
        station.dac,
        station.dac.ch01,
        gate_id=0,
        label="test_label",
        inter_delay=0,
        post_delay=0,
        max_voltage_step=0.01,
        ramp_rate=0.2,
    )
    yield gate


@pytest.fixture(scope="function")
def gate_2(station):
    gate = DeviceChannel(
        station.dac,
        station.dac.ch02,
        gate_id=1,
        label="test_label_2",
        inter_delay=0,
        post_delay=0,
        max_voltage_step=0.01,
        ramp_rate=0.2,
    )
    yield gate


@pytest.fixture(scope="function")
def gatecharacterization1D_settings(pinchoff_dmm, gate_1, tmp_path):
    pinchoff_dmm.po_current.gate = gate_1
    pinchoff_dmm.po_sensor.gate = gate_1

    readout_methods = ReadoutMethods(
        transport=pinchoff_dmm.po_current,
        sensing=pinchoff_dmm.po_sensor
    )

    setpoint_settings = SetpointSettings(
        voltage_precision=0.001,
        parameters_to_sweep=[gate_1.voltage],
        ranges_to_sweep=[[-0.1, 0]],
        safety_voltage_ranges=[(-3, 0)],
    )

    data_settings = DataSettings(
        db_name="temp.db",
        normalization_constants=NormalizationConstants(
            transport=(0, 1.2),
            sensing=(-0.13, 1.1),
            rf=(0, 1),
        ),
        db_folder=tmp_path,
    )

    tuningstage_settings = {
        "readout": readout_methods,
        "setpoint_settings": setpoint_settings,
        "data_settings": data_settings,
    }
    yield tuningstage_settings


@pytest.fixture(scope="function")
def chargediagram_settings(dot_dmm, tmp_path, gate_1, gate_2):

    dot_dmm.dot_current.gate_x = gate_1
    dot_dmm.dot_current.gate_y = gate_2

    dot_dmm.dot_sensor.gate_x = gate_1
    dot_dmm.dot_sensor.gate_y = gate_2

    readout_methods = ReadoutMethods(
        transport=dot_dmm.dot_current,
        sensing=dot_dmm.dot_sensor
    )

    setpoint_settings = SetpointSettings(
        voltage_precision=0.001,
        parameters_to_sweep=[gate_1.voltage, gate_2.voltage],
        ranges_to_sweep=[[-0.2, -0.1],[-0.3, -0.2]],
        safety_voltage_ranges=[(-3, 0), (-3, 0)],
    )

    data_settings = DataSettings(
        db_name="temp.db",
        normalization_constants=NormalizationConstants(
            transport=(0, 2),
            sensing=(-0.32, 3),
            rf=(0, 1),
        ),
        db_folder=tmp_path,
        segment_db_name="temp_dots_seg.db",
        segment_db_folder=tmp_path,
        segment_size=0.05,
    )

    tuningstage_settings = {
        "readout": readout_methods,
        "setpoint_settings": setpoint_settings,
        "data_settings": data_settings,
    }
    yield tuningstage_settings


@pytest.fixture(scope="function")
def chip_config():
    return os.path.join(PARENT_DIR, "device", "chip.yaml")


@pytest.fixture(scope="function")
def device(station, chip_config):
    if hasattr(station, "device_on_chip"):
        return station.device_on_chip

    station.load_config_file(chip_config)
    _chip = station.load_device_on_chip(station=station)
    return _chip


@pytest.fixture(scope="function")
def sim_station(station):
    station.add_component(
        MockDoubleQuantumDotInstrument('qd_mock_instrument')
    )
    try:
        yield station
    finally:
        station.close_all_registered_instruments()


@pytest.fixture(scope="function")
def sim_device(sim_station, chip_config):
    if hasattr(sim_station, "sim_device"):
        return sim_station.sim_device

    sim_station.load_config_file(chip_config)
    _dev = sim_station.load_sim_device(station=sim_station)
    return _dev


@pytest.fixture(scope="function")
def nanotune_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(nt.__file__)))


@pytest.fixture(scope="function")
def tuning_data_path():
    path = os.path.join(
        os.path.dirname(os.path.abspath(nt.__file__)), "..", "data", "tuning"
    )
    return path


@pytest.fixture(scope="function")
def sim_device_pinchoff(tuning_data_path, sim_station, sim_device):
    db_path = os.path.join(tuning_data_path, "device_characterization.db")
    qd_mock_instrument = sim_station.qd_mock_instrument

    pinchoff_data = QcodesDataProvider(
        [qd_mock_instrument.mock_device.right_plunger],
        db_path, "GB_Newtown_Dev_3_2", 1206)
    # Configure the simulator's drain pin to use the backing data
    qd_mock_instrument.mock_device.drain.set_data_provider(
        pinchoff_data)

    sim_device.right_plunger.voltage = qd_mock_instrument.right_plunger
    sim_device.all_gates_to_highest()
    ds = nt.Dataset(
        1206,
        "device_characterization.db",
        db_folder=tuning_data_path
    )
    sim_device.normalization_constants = ds.normalization_constants
    return sim_device


@pytest.fixture(scope="function")
def pinchoff_classifier(nanotune_path):
    _pinchoff_classifier = Classifier(
        ['pinchoff.npy'],
        'pinchoff',
        data_types=["signal"],
        classifier="MLPClassifier",
        folder_path=os.path.join(nanotune_path, 'data', 'training_data'),
    )
    _pinchoff_classifier.train()
    return _pinchoff_classifier


@pytest.fixture(scope="function")
def tuner_default_input(tmp_path):
    settings = {
        "name": "test_tuner",
        "data_settings": DataSettings(
            db_name="temp.db",
            db_folder=str(tmp_path),
        ),
        "classifiers": Classifiers(pinchoff=MockClassifer(category="pinchoff")),
        "setpoint_settings": SetpointSettings(
            voltage_precision=0.001,
            ranges_to_sweep=[(-1, 0)],
            safety_voltage_ranges=[(-3, 0)],
        ),
    }
    yield settings


@pytest.fixture(scope="function")
def tuner(tuner_default_input):
    tuner = Tuner(**tuner_default_input)
    try:
        yield tuner
    finally:
        tuner.close()
