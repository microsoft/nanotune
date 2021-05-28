import json

from qcodes.dataset.measurements import Measurement

import nanotune as nt


def save_1Ddata_with_qcodes(data_generator_method, metadata_generator_method):
    meas = Measurement()
    meas.register_custom_parameter(
        "voltage", paramtype="numeric", label="voltage x", unit="V"
    )
    meas.register_custom_parameter(
        "current",
        paramtype="numeric",
        label="dc current",
        unit="A",
        setpoints=("voltage",),
    )

    meas.register_custom_parameter(
        "sensor",
        paramtype="numeric",
        label="dc sensor",
        unit="A",
        setpoints=("voltage",),
    )
    voltage, current, sensor = data_generator_method()
    with meas.run() as datasaver:
        datasaver.add_result(("voltage", voltage), ("current", current))
        datasaver.add_result(("voltage", voltage), ("sensor", sensor))

        if metadata_generator_method is not None:
            nt_metadata, current_label = metadata_generator_method()
            datasaver.dataset.add_metadata(nt.meta_tag, json.dumps(nt_metadata))
            datasaver.dataset.add_metadata("snapshot", json.dumps({}))
            for label, value in current_label.items():
                datasaver.dataset.add_metadata(label, value)

    return datasaver


def save_2Ddata_with_qcodes(data_generator_method, metadata_generator_method):
    meas = Measurement()
    meas.register_custom_parameter(
        "v_x", paramtype="numeric", label="voltage x", unit="V"
    )
    meas.register_custom_parameter(
        "v_y", paramtype="numeric", label="voltage y", unit="V"
    )
    meas.register_custom_parameter(
        "current",
        paramtype="numeric",
        label="dc current",
        unit="A",
        setpoints=("v_x", "v_y"),
    )
    meas.register_custom_parameter(
        "sensor",
        paramtype="numeric",
        label="dc sensor",
        unit="A",
        setpoints=("v_x", "v_y"),
    )

    with meas.run() as datasaver:
        xv, yv, ddot, sensor = data_generator_method()
        datasaver.add_result(("v_x", xv), ("v_y", yv), ("current", ddot))
        datasaver.add_result(("v_x", xv), ("v_y", yv), ("sensor", sensor))

        datasaver.dataset.add_metadata("snapshot", json.dumps({}))
        if metadata_generator_method is not None:
            nt_metadata, current_label = metadata_generator_method()

            datasaver.dataset.add_metadata(nt.meta_tag, json.dumps(nt_metadata))
            for label, value in current_label.items():
                datasaver.dataset.add_metadata(label, value)

    return datasaver
