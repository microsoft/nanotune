import pytest
from nanotune.device_tuner.tuner import linear_voltage_steps
from nanotune.device_tuner.tuningresult import MeasurementHistory
from nanotune.tuningstages.settings import Classifiers


def test_measure_initial_ranges_2D(
    tuner,
    sim_device_playback,
    sim_scenario_init_ranges,
    pinchoff_classifier,
):
    self = tuner
    self.classifiers = Classifiers(pinchoff=pinchoff_classifier)
    device = sim_device_playback
    gate_to_set = device.top_barrier
    voltage_step = 0.2
    scenario = sim_scenario_init_ranges

    gates_to_sweep = [
        device.central_barrier,
        device.left_barrier, device.right_barrier]

    if self.classifiers.pinchoff is None:
        raise KeyError("No pinchoff classifier.")
    device.all_gates_to_highest()

    v_steps = linear_voltage_steps(
        gate_to_set.safety_voltage_range(), voltage_step)

    # using `get_pairwise_pinchoff`, code pasted in:
    measurement_result = MeasurementHistory(device.name)
    layout_ids = [g.gate_id for g in gates_to_sweep]
    skip_gates = dict.fromkeys(layout_ids, False)

    for last_voltage in v_steps:
        gate_to_set.voltage(last_voltage)

        for gate in gates_to_sweep:
            if not skip_gates[gate.gate_id]:
                scenario.run_next_step()
                sub_tuning_result = self.characterize_gates(
                    device,
                    [gate],
                    use_safety_voltage_ranges=True,
                    comment=f"Measuring initial range of {gate.full_name} \
                        with {gate_to_set.full_name} at {last_voltage}."
                )
                measurement_result.add_result(sub_tuning_result)
                if sub_tuning_result.success:
                    skip_gates[gate.gate_id] = True
                    last_gate_to_pinchoff = gate

        if all(skip_gates.values()):
            break

    min_voltage = last_voltage

    assert min_voltage == -0.6428571428571428
    assert last_gate_to_pinchoff == device.central_barrier

    device.all_gates_to_highest()
    # Swap top_barrier and last barrier to pinch off agains it to
    # determine opposite corner of valid voltage space.
    v_steps = linear_voltage_steps(
        last_gate_to_pinchoff.safety_voltage_range(), voltage_step)

    gates_to_sweep = [gate_to_set]
    gate_to_set = last_gate_to_pinchoff

    # using `get_pairwise_pinchoff` code pasted in:
    measurement_result2 = MeasurementHistory(device.name)
    layout_ids = [g.gate_id for g in gates_to_sweep]
    skip_gates = dict.fromkeys(layout_ids, False)

    for last_voltage in v_steps:
        gate_to_set.voltage(last_voltage)

        for gate in gates_to_sweep:
            if not skip_gates[gate.gate_id]:
                scenario.run_next_step()
                sub_tuning_result = self.characterize_gates(
                    device,
                    [gate],
                    use_safety_voltage_ranges=True,
                    comment=f"Measuring initial range of {gate.full_name} \
                        with {gate_to_set.full_name} at {last_voltage}."
                )
                measurement_result2.add_result(sub_tuning_result)
                if sub_tuning_result.success:
                    skip_gates[gate.gate_id] = True
                    last_gate_to_pinchoff = gate

        if all(skip_gates.values()):
            break

    # The line below is correct with max_voltage = ... "low_voltage"
    features = measurement_result2.last_added.ml_result["features"]
    max_voltage = features["transport"]["low_voltage"]

    assert max_voltage == -2.918972990997

    measurement_result.update(measurement_result2)
    assert len(measurement_result.to_dict()) == 14

    device.all_gates_to_highest()

