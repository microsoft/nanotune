# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import time

import matplotlib.pyplot as plt
import pytest
from functools import partial

import nanotune as nt
from nanotune.device.device import ReadoutMethods, NormalizationConstants
from nanotune.fit.pinchofffit import PinchoffFit
from nanotune.tests.mock_classifier import MockClassifer
from nanotune.tuningstages.base_tasks import *


def test_save_machine_learning_result(qc_dataset_doubledot):
    run_id = 1

    save_machine_learning_result(
        run_id,
        {"quality": True},
    )
    ds = load_by_id(run_id)
    nt_meta = json.loads(ds.get_metadata(nt.meta_tag))

    assert bool(nt_meta["predicted_quality"])

    assert "predicted_regime" not in nt_meta.keys()
    save_machine_learning_result(
        run_id,
        {"predicted_regime": "doubledot"},
    )
    ds = load_by_id(run_id)
    nt_meta = json.loads(ds.get_metadata(nt.meta_tag))
    assert "predicted_predicted_regime" not in nt_meta.keys()
    assert nt_meta["predicted_regime"] == "doubledot"


def test_check_measurement_quality(experiment, tmp_path):
    clf = MockClassifer("pinchoff")
    assert check_measurement_quality(clf, 1, "temp.db", tmp_path)


def test_save_extracted_features(qc_dataset_pinchoff, tmp_path):
    run_id = 1
    save_extracted_features(
        PinchoffFit,
        run_id,
        "temp.db",
        db_folder=tmp_path,
    )

    ds = load_by_id(run_id)
    nt_meta = json.loads(ds.get_metadata(nt.meta_tag))
    assert isinstance(nt_meta["features"], dict)


def test_get_measurement_features(nt_dataset_pinchoff, tmp_path):
    run_id = 1
    features = get_measurement_features(
        run_id,
        "temp.db",
        tmp_path,
    )
    ds = nt.Dataset(run_id, "temp.db", db_folder=tmp_path)
    assert ds.features == features


def test_set_up_gates_for_measurement(gate_1, gate_2):
    setpoints = [[-0.2, -0.15, -0.1], [-0.1, -0.15, -0.2]]
    gate_1.use_ramp(True)
    gate_2.use_ramp(True)

    gate_1.voltage(0)
    gate_2.voltage(0)

    with set_up_gates_for_measurement(
        [gate_1.voltage, gate_2.voltage],
        setpoints,
    ) as setup:
        assert not gate_1.use_ramp()
        assert not gate_2.use_ramp()

        assert gate_1.voltage() == -0.2
        assert gate_2.voltage() == -0.1

    assert gate_1.use_ramp()
    assert gate_2.use_ramp()


def test_set_post_delay(gate_1, gate_2):
    gate_1.voltage.post_delay = 0
    gate_2.voltage.post_delay = 0

    set_post_delay([gate_1.voltage, gate_2.voltage], 0.2)
    assert gate_1.voltage.post_delay == 0.2
    assert gate_2.voltage.post_delay == 0.2

    set_post_delay([gate_1.voltage, gate_2.voltage], [0.1, 0.3])
    assert gate_1.voltage.post_delay == 0.1
    assert gate_2.voltage.post_delay == 0.3


def test_swap_range_limits_if_needed():
    current_voltages = [-0.12, -0.18]

    new_ranges = swap_range_limits_if_needed(
        current_voltages, [(-0.3, -0.1), (-0.2, -0.1)]
    )
    assert new_ranges[0] == (-0.1, -0.3)
    assert new_ranges[1] == (-0.2, -0.1)


def test_compute_linear_setpoints():
    setpoints = compute_linear_setpoints([(-0.3, -0.2), (-0.5, -0.1)], 0.01)
    assert len(setpoints) == 2
    assert (setpoints[0] == np.linspace(-0.3, -0.2, 10)).all()


def test_prepare_metadata(dummy_dmm):
    metadata = prepare_metadata(
        "test_device",
        NormalizationConstants(transport=(0, 1.2)),
        ReadoutMethods(transport=dummy_dmm.dac1),
    )

    assert metadata["normalization_constants"] == {
        "transport": (0, 1.2), "sensing": (0., 1.), "rf": (0., 1.)}
    assert metadata["device_name"] == "test_device"
    assert metadata["readout_methods"] == {
        "transport": dummy_dmm.dac1.full_name}
    assert "git_hash" in metadata.keys()
    assert "features" in metadata.keys()


def test_add_metadata_to_dict(dummy_dmm):
    meta_dict = {"device_name": "test_device"}
    new_dict = add_metadata_to_dict(
        meta_dict,
        {
            "features": {"amplitude": 0.2},
            "readout_methods": {"transport": dummy_dmm.dac1},
        },
    )
    assert new_dict["features"] == {"amplitude": 0.2}
    assert new_dict["readout_methods"]["transport"] == dummy_dmm.dac1


def test_save_metadata(qc_dataset_pinchoff, tmp_path):
    run_id = 1

    ds = load_by_id(run_id)
    initial_metadata = {"features": {"amplitude": 0.2}}
    ds.add_metadata(nt.meta_tag, json.dumps(initial_metadata))

    save_metadata(
        run_id,
        {"device_name": "test_device"},
        nt.meta_tag,
    )
    ds = load_by_id(run_id)
    metadata = json.loads(ds.get_metadata(nt.meta_tag))
    assert metadata["device_name"] == "test_device"
    assert metadata["features"] == initial_metadata["features"]

    assert not ["features", "device_name"] - metadata.keys()


def test_get_elapsed_time():
    elapsed_time, formatted_time = get_elapsed_time(
        time.time(),
        time.time(),
    )
    assert isinstance(elapsed_time, float)
    assert isinstance(formatted_time, str)


def test_plot_fit(nt_dataset_pinchoff, tmp_path):
    plot_fit(PinchoffFit, 1, "temp.db", db_folder=tmp_path)
    plt.close()


def test_print_tuningstage_status(capsys):
    tuning_result = TuningResult(
        'pinchoff',
        True,
        termination_reasons=['device pinched off'],
        data_ids=[1],
        ml_result={'regime': 'pinchoff'},
        timestamp="",
    )
    print_tuningstage_status(tuning_result)
    captured = capsys.readouterr()

    assert 'good' in captured.out.lower()
    assert 'pinchoff' in captured.out.lower()
    assert 'device pinched off' in captured.out.lower()


def test_take_data_add_metadata(gate_1, gate_2, dummy_dmm, experiment):
    gate_1.max_voltage_step = 0.1
    gate_2.max_voltage_step = 0.1
    params_to_sweep = [gate_1.voltage, gate_2.voltage]
    params_to_measure = [dummy_dmm.dac1]
    setpoints = [
        list(np.linspace(-0.3, -0.1, 10)), list(np.linspace(-0.2, -0.1, 10)),
    ]
    pre_measurement_metadata = {
        'device_name': 'test_sample',
        'normalization_constants': {'transport': [0, 1.2]},
        'readout_methods': {'transport': dummy_dmm.dac1.full_name},
    }
    run_id = take_data_add_metadata(
        params_to_sweep,
        params_to_measure,
        setpoints,
        pre_measurement_metadata,
    )

    dataset = load_by_id(run_id)
    metadata = json.loads(dataset.get_metadata(nt.meta_tag))
    print(metadata)

    assert 'elapsed_time' in metadata
    assert metadata['device_name'] == 'test_sample'
    assert metadata['normalization_constants']['transport'] == [0, 1.2]
    assert metadata['readout_methods']['transport'] == dummy_dmm.dac1.full_name


def test_run_stage(experiment, gate_1, gate_2, dummy_dmm):

    params_to_sweep = [gate_1.voltage, gate_2.voltage]
    params_to_measure = [dummy_dmm.dac1]
    ml_result = {'regime': 'pinchoff', 'quality': True}

    compute_setpoint_task = partial(
        compute_linear_setpoints, voltage_precision=0.1,
    )

    tuning_result = run_stage(
        'pinchoff',
        params_to_sweep,
        params_to_measure,
        [(-0.3, 0), (-0.3, 0)],
        compute_setpoint_task,
        take_data,
        lambda x: ml_result,
        save_machine_learning_result,
        lambda x: True,
    )

    assert tuning_result.success
    assert tuning_result.stage == 'pinchoff'
    assert not tuning_result.termination_reasons
    assert tuning_result.ml_result == ml_result
    assert tuning_result.data_ids == [1]


def test_iterate_stage(experiment, gate_1, gate_2, dummy_dmm):

    params_to_sweep = [gate_1.voltage, gate_2.voltage]
    params_to_measure = [dummy_dmm.dac1]
    compute_setpoint_task = partial(
        compute_linear_setpoints, voltage_precision=0.1,
    )
    ml_result = {'regime': 'pinchoff', 'quality': True}
    run_stage_tasks = [
        compute_setpoint_task,
        take_data,
        lambda x: ml_result,
        save_machine_learning_result,
        lambda x: True,
    ]
    def conclude_iteration(a, b, c, d, e):
        return True, [(-0.3, 0), (-0.3, 0)], []
    def display_result(a, b):
        return None

    tuning_result = iterate_stage(
        'pinchoff',
        params_to_sweep,
        params_to_measure,
        [(-0.3, 0), (-0.3, 0)],
        [(-3, 0), (-3, 0)],
        run_stage,
        run_stage_tasks,
        conclude_iteration,
        display_result,
    )

    assert tuning_result.ml_result == ml_result
    assert tuning_result.data_ids == [1]
    assert tuning_result.termination_reasons == []

    # test multiple iterations
    def conclude_iteration(a, b, c, current_iteration, max_n_iterations):
        if current_iteration < max_n_iterations:
            return False, [(-0.3, 0), (-0.3, 0)], ['not done yet']
        else:
            return True, [(-0.3, 0), (-0.3, 0)], ['not done yet']

    tuning_result = iterate_stage(
        'pinchoff',
        params_to_sweep,
        params_to_measure,
        [(-1, 0), (-1, 0)],
        [(-3, 0), (-3, 0)],
        run_stage,
        run_stage_tasks,
        conclude_iteration,
        display_result,
        max_n_iterations=2,
    )

    assert tuning_result.ml_result == ml_result
    assert tuning_result.data_ids == [2, 3]
    assert tuning_result.termination_reasons == ['not done yet']

def test_conclude_iteration_with_range_update():

    tuning_result = TuningResult(
        'pinchoff',
        False,
        termination_reasons=['device pinched off'],
        data_ids=[1],
        ml_result={'regime': 'pinchoff'},
        timestamp="",
    )
    def get_range_update_directives(*args , **kwargs):
        return ['x more negative'], ['open current']

    def get_new_current_ranges(
        current_valid_ranges,
        safety_voltage_ranges,
        range_update_directives):
        return [(-0.3, 0), (-0.1, 0)]

    # no successful tuning result and new voltage ranges returned:
    (done,
     new_voltage_ranges,
     termination_reasons) = conclude_iteration_with_range_update(
        tuning_result,
        [(-0.1, 0), (-0.1, 0)],  # current_valid_ranges
        [(-3, 0), (-3, 0)],  # safety_voltage_ranges,
        get_range_update_directives,
        get_new_current_ranges,
        1,  # current_iteration,
        2,  # max_n_iterations,
    )
    assert not done
    assert new_voltage_ranges == [(-0.3, 0), (-0.1, 0)]
    assert termination_reasons == ['open current']

    # no successful tuning result but no range_update_directives:
    def get_range_update_directives2(*args , **kwargs):
        return [], ['safety ranges reached']

    (done,
     new_voltage_ranges,
     termination_reasons) = conclude_iteration_with_range_update(
        tuning_result,
        [(-0.1, 0), (-0.1, 0)],  # current_valid_ranges
        [(-3, 0), (-3, 0)],  # safety_voltage_ranges,
        get_range_update_directives2,
        get_new_current_ranges,
        1,  # current_iteration,
        2,  # max_n_iterations,
    )
    assert done
    assert new_voltage_ranges == []
    assert termination_reasons == ['safety ranges reached']

    # max iteration reached:
    (done,
     new_voltage_ranges,
     termination_reasons) = conclude_iteration_with_range_update(
        tuning_result,
        [(-0.1, 0), (-0.1, 0)],  # current_valid_ranges
        [(-3, 0), (-3, 0)],  # safety_voltage_ranges,
        get_range_update_directives,
        get_new_current_ranges,
        1,  # current_iteration,
        1,  # max_n_iterations,
    )
    assert done
    assert 'max_n_iterations reached' in termination_reasons

    # successful tuning result found:
    tuning_result.success = True
    (done,
     new_voltage_ranges,
     termination_reasons) = conclude_iteration_with_range_update(
        tuning_result,
        [(-0.1, 0), (-0.1, 0)],  # current_valid_ranges
        [(-3, 0), (-3, 0)],  # safety_voltage_ranges,
        get_range_update_directives,
        get_new_current_ranges,
        1,  # current_iteration,
        2,  # max_n_iterations,
    )
    assert done
    assert termination_reasons == []

def test_get_current_voltages(gate_1, gate_2):
    gate_1.voltage(-0.4)
    gate_2.voltage(-0.6)
    voltages = get_current_voltages([gate_1.voltage, gate_2.voltage])
    assert voltages == [-0.4, -0.6]

def test_set_voltages(gate_1, gate_2):
    gate_1.voltage(-0.4)
    gate_2.voltage(-0.6)

    set_voltages([gate_1.voltage, gate_2.voltage], [-0.5, -0.7])
    assert gate_1.voltage() == -0.5
    assert gate_2.voltage() == -0.7

def test_get_fit_range_update_directives(db_real_pinchoff, tmp_path):
    directives = get_fit_range_update_directives(
        PinchoffFit,
        345,
        'pinchoff_data.db',
        tmp_path,
    )
    assert directives == ['x more negative']
