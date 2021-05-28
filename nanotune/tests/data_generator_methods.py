import numpy as np
import qcodes as qc

import nanotune as nt
from nanotune.math.gaussians import gaussian2D_fct
from nanotune.math.lorentzians import lorentzian_1D
from nanotune.tests.data_savers import (save_1Ddata_with_qcodes,
                                        save_2Ddata_with_qcodes)

META_FIELDS = nt.config["core"]["meta_fields"]
NT_LABELS = list(dict(nt.config["core"]["labels"]).keys())
test_data_labels = {
    0: "pinchoff",
    1: "pinchoff",
    2: "singledot",
    3: None,
    4: "doubledot",
    5: "pinchoff",
    6: None,
    7: None,
    8: "doubledot",
    9: None,
}


class DotCurrent(qc.Parameter):
    def __init__(self, name, gate_x=None, gate_y=None, **kwargs):
        super().__init__(name, **kwargs)
        self._cr = self._current()
        self.gate_x = gate_x
        self.gate_y = gate_y
        next(self._cr)

    def get_raw(self):
        val = self._cr.send((self.gate_x.dc_voltage(), self.gate_y.dc_voltage()))
        next(self._cr)
        return val

    def _current(self):
        x, y = 0, 0
        while True:
            x, y = yield
            yield doubledot_triple_points(x, y)


class DotSensor(qc.Parameter):
    def __init__(self, name, gate_x=None, gate_y=None, **kwargs):
        super().__init__(name, **kwargs)
        self._se = self._sensor()
        self.gate_x = gate_x
        self.gate_y = gate_y
        next(self._se)

    def get_raw(self):
        val = self._se.send((self.gate_x.dc_voltage(), self.gate_y.dc_voltage()))
        next(self._se)
        return val

    def _sensor(self):
        x, y = 0, 0
        while True:
            x, y = yield
            val = doubledot_triple_points(x, y)
            val += np.random.normal(0, 1, 1) * 0.2
            yield val


class PinchoffCurrent(qc.Parameter):
    def __init__(self, name, gate=None, **kwargs):
        super().__init__(name, **kwargs)
        self.gate = gate
        self._cr = self._current()
        next(self._cr)

    def get_raw(self):
        val = self._cr.send(self.gate.dc_voltage())
        next(self._cr)
        return val

    def _current(self):
        """ Yields pinchoff current"""
        x = 0
        while True:
            x = yield
            yield pinchoff_curve(x)


class PinchoffSensor(qc.Parameter):
    def __init__(self, name, gate=None, **kwargs):
        super().__init__(name, **kwargs)
        self._se = self._sensor()
        self.gate = gate
        next(self._se)

    def get_raw(self):
        val = self._se.send(self.gate.dc_voltage())
        next(self._se)
        return val

    def _sensor(self):
        """ Yields pinchoff current"""
        x = 0
        while True:
            x = yield
            curve = pinchoff_curve(x)
            curve += np.random.normal(0, 1, 1) * 0.1
            yield curve


def pinchoff_curve(voltage):
    return 0.6 * (1 + np.tanh(1000 * voltage + 50))


def generate_pinchoff_data():
    voltage = np.linspace(-0.1, 0, 120)
    current = pinchoff_curve(voltage)

    np.random.seed(0)
    noise = np.random.normal(0, 1, 120) * 0.05
    sensor = 0.5 * pinchoff_curve(voltage) / 0.6
    sensor += noise

    return voltage, current, sensor


def generate_bad_pinchoff_data():
    voltage = np.linspace(-0.1, 0, 120)
    current = 0.2 * (1 + np.tanh(700 * voltage + 50)) + 0.3

    np.random.seed(0)
    noise = np.random.normal(0, 1, 120) * 0.05
    sensor = 0.1 * (1 + np.tanh(700 * voltage + 50)) + 0.1
    sensor += noise

    return voltage, current, sensor


def generate_pinchoff_metadata():
    current_label = dict.fromkeys(NT_LABELS, 0)
    current_label["pinchoff"] = 1
    current_label["good"] = 1

    nt_metadata = dict.fromkeys(META_FIELDS, None)
    nt_metadata["device_name"] = "test_device"
    nt_metadata["normalization_constants"] = {
        "dc_current": [0, 1.2],
        "dc_sensor": [-0.13, 1.1],
        "rf": [0, 1],
    }
    nt_metadata["device_max_signal"] = 1.2
    nt_metadata["readout_methods"] = {"dc_current": "current", "dc_sensor": "sensor"}
    nt_metadata["features"] = {
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

    return nt_metadata, current_label


def doubledot_triple_points(xv, yv):
    gauss = gaussian2D_fct(1, -0.19, -0.25, 0.01, 0.01)
    gauss2 = gaussian2D_fct(1, -0.17, -0.22, 0.01, 0.01)
    gauss3 = gaussian2D_fct(1, -0.14, -0.28, 0.01, 0.01)
    gauss4 = gaussian2D_fct(1, -0.12, -0.25, 0.01, 0.01)

    ddot = gauss(xv, yv) + gauss2(xv, yv) + gauss3(xv, yv) + gauss4(xv, yv) * 1.4
    return ddot


def generate_doubledot_data():
    x = np.linspace(-0.2, -0.1, 81)
    y = np.linspace(-0.3, -0.2, 51)

    xv, yv = np.meshgrid(x, y)
    ddot = doubledot_triple_points(xv, yv)

    np.random.seed(0)
    noise = np.random.normal(0, 1, np.prod(ddot.shape)) * 0.1
    sensor = ddot + np.resize(noise, ddot.shape)

    return xv, yv, ddot, sensor


def generate_doubledot_metadata():
    current_label = dict.fromkeys(NT_LABELS, 0)
    current_label["doubledot"] = 1
    current_label["good"] = 1

    META_FIELDS = nt.config["core"]["meta_fields"]
    nt_metadata = dict.fromkeys(META_FIELDS, None)
    nt_metadata["device_name"] = "test_device"
    nt_metadata["normalization_constants"] = {}
    nt_metadata["normalization_constants"]["dc_current"] = [0, 2]
    nt_metadata["normalization_constants"]["dc_sensor"] = [-0.32, 3]
    nt_metadata["normalization_constants"]["rf"] = [0, 1]
    nt_metadata["device_max_signal"] = 2
    nt_metadata["readout_methods"] = {"dc_current": "current", "dc_sensor": "sensor"}
    nt_metadata["features"] = {
        "dc_current": {"triple_points": None},
        "dc_sensor": {"triple_points": None},
    }

    return nt_metadata, current_label


def generate_default_metadata():
    nt_metadata = dict.fromkeys(META_FIELDS, None)
    nt_metadata["device_name"] = "test_device"
    nt_metadata["normalization_constants"] = {
        key: (0, 1) for key in ["dc_current", "rf", "dc_sensor"]
    }
    nt_metadata["device_max_signal"] = 1
    nt_metadata["readout_methods"] = {"dc_current": "current"}

    current_label = dict.fromkeys(NT_LABELS, 0)
    return nt_metadata, current_label


def generate_coulomboscillations():
    v = np.linspace(-1, -0.5, 120)

    l1 = lorentzian_1D(v, 1, -0.9, 0.02)
    l2 = lorentzian_1D(v, 1.4, -0.8, 0.02)
    l3 = lorentzian_1D(v, 1.1, -0.6, 0.02)

    current = l1 + l2 + l3

    s1 = lorentzian_1D(v, 0.8, -0.95, 0.02)
    s2 = lorentzian_1D(v, 1, -0.85, 0.02)
    s3 = lorentzian_1D(v, 0.9, -0.65, 0.02)
    sensor = s1 + s2 + s3
    np.random.seed(0)
    noise = np.random.normal(0, 1, 120) * 0.03
    sensor = sensor + noise

    return v, current, sensor


def generate_coloumboscillation_metadata():
    current_label = dict.fromkeys(NT_LABELS, 0)
    current_label["coulomboscillation"] = 1
    current_label["good"] = 1

    nt_metadata = dict.fromkeys(META_FIELDS, None)
    nt_metadata["device_name"] = "test_device"
    nt_metadata["normalization_constants"] = {}
    nt_metadata["normalization_constants"]["dc_current"] = [0.013, 1.4]
    nt_metadata["normalization_constants"]["dc_sensor"] = [-0.03, 1.1]
    nt_metadata["normalization_constants"]["rf"] = [0, 1]
    nt_metadata["device_max_signal"] = 1.4
    nt_metadata["readout_methods"] = {"dc_current": "current", "dc_sensor": "sensor"}
    nt_metadata["features"] = {}

    return nt_metadata, current_label


def populate_db_doubledots():
    for _ in range(len(test_data_labels)):
        _ = save_2Ddata_with_qcodes(
            generate_doubledot_data, generate_doubledot_metadata
        )


def populate_db_pinchoffs(n_datasets: int = 10):
    for _ in range(n_datasets):
        _ = save_1Ddata_with_qcodes(generate_pinchoff_data, generate_pinchoff_metadata)


def populate_db_coulomboscillations():
    for _ in range(10):
        _ = save_1Ddata_with_qcodes(
            generate_coulomboscillations, generate_coloumboscillation_metadata
        )
