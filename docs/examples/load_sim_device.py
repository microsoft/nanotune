# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

def load_sim_device(station, chip_config_path):
    if hasattr(station, "sim_device"):
        return station.sim_device

    station.load_config_file(chip_config_path)
    dev = station.load_sim_device(station=station)
    qd_mock_instrument = station.qd_mock_instrument
    dev.top_barrier.voltage = qd_mock_instrument.top_barrier
    dev.top_barrier.inter_delay = 0
    dev.left_barrier.voltage = qd_mock_instrument.left_barrier
    dev.left_barrier.inter_delay = 0
    dev.left_plunger.voltage = qd_mock_instrument.left_plunger
    dev.left_plunger.inter_delay = 0
    dev.central_barrier.voltage = qd_mock_instrument.central_barrier
    dev.central_barrier.inter_delay = 0
    dev.right_plunger.voltage = qd_mock_instrument.right_plunger
    dev.right_plunger.inter_delay = 0
    dev.right_barrier.voltage = qd_mock_instrument.right_barrier
    dev.right_barrier.inter_delay = 0

    dev.initial_valid_ranges({0: [-0.9, -0.6]})
    dev.current_valid_ranges({0: [-0.9, -0.6]})

    return dev
