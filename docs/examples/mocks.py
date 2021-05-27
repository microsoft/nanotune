import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import ChannelList, InstrumentChannel
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import Parameter


PINCH_OFF_FILE = "mock_data/pinch_off.csv"


class PinchOffCurrentParameter(Parameter):
    """Virtual parameter for simulating a pinch-off curve"""
    def __init__(
        self,
        name: str,
        setter_param: Parameter,
        acq_delay: float = 0.01,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._acq_delay = acq_delay
        self._data = pd.read_csv(PINCH_OFF_FILE)
        # Interpolate/extrapolate real data to get emulated version of pinch-off curve
        self.pinch_off_fcn = interp1d(
            x=self._data[setter_param.name],
            y=self._data[name],
            bounds_error=False,
            fill_value="extrapolate"
        )
        self._po = self._pinch_off()
        self.setter_param = setter_param
        next(self._po)

    def get_raw(self):
        """
        This method is automatically wrapped to
        provide a ``get`` method on the parameter instance.
        """
        val = self._po.send(self.setter_param())
        next(self._po)
        return val

    def _pinch_off(self):
        """
        Yields pinch off current for a given gate voltage
        """
        gate_voltage = 0
        while True:
            gate_voltage = yield
            yield self.pinch_off_fcn(gate_voltage)


class TopgatedABRingParameter(PinchOffCurrentParameter):
    """Virtual parameter for simulating an AB ring with a top gate"""
    def __init__(
        self,
        name: str,
        setter_param: Parameter,
        field: Parameter,
        field_step: float = 1e-4,
        **kwargs
    ):
        super().__init__(name, setter_param=setter_param, **kwargs)
        self._ab_params = {"dB": 4e-3, "slope": .1}
        self._tp = self._transport()
        self.field = field
        self._field_step = field_step
        self._last_field = field()
        self._generated_field = None
        self._last_dfield = 1.
        self.reset_phase()
        next(self._tp)

    @staticmethod
    def _generate_ab_distribution(scales=(0.1, 0.02), n=100):
        """Sample a normal distribution to get simulated signal for AB oscillations"""
        _dB_norms = []
        for scale in scales:
            _dB_norms += [_dB_norm for _dB_norm in np.random.normal(loc=1., scale=scale, size=n)]
        return _dB_norms

    def reset_phase(self):
        self._phase = np.random.rand() * np.pi

    @staticmethod
    def noise_params_fcn(voltage: float, scaling: float = 2e-2):
        """Generate noise parameters A1 (constant noise) and A2 (Gaussian noise)"""
        A0 = 5.
        if voltage > .5:
            A1 = 2.
            A2 = .8
        elif voltage <= -0.5:
            A1 = 0.2
            A2 = 0.2
        else:
            A1 = 1. + voltage/.5
            A2 = .4 + .4 * np.abs(voltage)/.5

        return A0 * scaling, A1 * scaling, A2 * scaling

    @property
    def phase(self):
        """Get the phase of the AB signal depending on the B-field sweep direction"""
        return self._phase

    def do_hysteresis(self, field: float):
        # Check for hysteresis
        dfield = field - self._last_field
        do_hysteresis = np.sign(self._last_dfield) != np.sign(dfield)

        # If asking same field twice in a row, don't switch phase
        if dfield == 0.:
            return False

        self._last_field, self._last_dfield = field, dfield
        return do_hysteresis

    def do_generate_data(self, field: float, window: float):
        if self._generated_field is None:
            return True
        return np.abs(field - self._generated_field) > window

    @staticmethod
    def ab_fcn(
        phase: float,
        A0: float,
        A1: float,
        A2: float,
        B: Iterable,
        dB: float=4e-3,
        dB_norms: Iterable[float]=None,
        slope: float=.1
    ) -> Iterable:
        """
        Simulated Aharonov-Bohm oscillations with noise and time window filtering
        using a Savitsky-Golay filter, see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
        """
        size = 1
        if isinstance(B, Iterable):
            size = len(B)

        def _fast(B, dB, phase, n=100):
            fast = 0
            for _dB_norm in dB_norms:
                _dB = _dB_norm * dB
                fast += np.sin(phase + B * 2 * np.pi / _dB)
            fast = fast / np.max(fast)
            return fast

        fast = _fast(B, dB, phase)

        slow1 = np.sin(phase + B * 2 * np.pi / 100e-3)
        slow2 = np.sin(phase + B * 2 * np.pi / 50e-3)
        slow = (slow1 + slow2) / 2.

        random1 = np.random.rand(size)
        random2 = np.random.normal(size=size)

        y = A0 * slow + (A1 * random1 + A2 * random2) * fast + slope * B
        yhat = savgol_filter(y, 11, 3)
        yhat = savgol_filter(yhat, 11, 3)
        return yhat

    def get_raw(self):
        """
        This method is automatically wrapped to
        provide a ``get`` method on the parameter instance.
        """
        time.sleep(self._acq_delay)
        val = self._tp.send((self.setter_param(), self.field()))
        next(self._tp)
        amp = super().get_raw() # amplitude by pinch off value

        return amp * (1 + val)

    def generate_data(self, field: float, voltage: float, window: float):
        """Generate B field sweep data"""
        self._generated_field = field
        B = np.arange(field - window, field + window, self._field_step)
        A0, A1, A2 = self.noise_params_fcn(voltage=voltage) # Get noise parameters
        dB_norms = self._generate_ab_distribution()
        signal = self.ab_fcn(phase=self.phase, A0=A0, A1=A1, A2=A2, B=B, dB_norms=dB_norms, **self._ab_params)
        self._ab_osc = interp1d(
            x=B,
            y=signal,
            bounds_error=False,
            fill_value="extrapolate"
        )
        self._signal = signal

    def _transport(self, window: float = .05):
        """
        Yields AB oscillations data for a given B field and gate voltage
        """
        last_gate_voltage = 0.0
        self.generate_data(field=self._last_field, voltage=last_gate_voltage, window=window)
        while True:
            (gate_voltage, field) = yield
            if self.do_hysteresis(field):# or gate_voltage != last_gate_voltage:
                self.reset_phase()
                self.generate_data(field=field, voltage=gate_voltage, window=window)
                last_gate_voltage = gate_voltage

            if self.do_generate_data(field, window=window):
                self.generate_data(field=field, voltage=gate_voltage, window=window)

            yield self._ab_osc(field)


class RFReadoutParameter(TopgatedABRingParameter):
    """Virtual parameter for simulating a pinch-off curve"""
    def __init__(
        self,
        name: str,
        setter_param: Parameter,
        field: Parameter,
        rf_source: Instrument,
        **kwargs
    ):
        super().__init__(name, setter_param=setter_param, field=field, **kwargs)
        self._sin_params = [0.8134999989710102, 0.0, -0.18650000102898978]
        self.phase_fcn = lambda x, a, b, c: a * np.sin(np.pi * (x - b) / 180.) + c
        self._ro = self._readout()
        self.rf_source = rf_source
        next(self._ro)

    def random_phase(self):
        self._sin_params[1] = np.random.rand() * 180.

    def get_raw(self):
        """
        This method is automatically wrapped to
        provide a ``get`` method on the parameter instance.
        """
        if self.rf_source is not None:
            if self.rf_source.status() is False:
                return 0.0
        val = self._ro.send(self.rf_source.phase())
        next(self._ro)
        amp = super().get_raw() # multiply signal by amplitude
        return val * amp

    def _readout(self):
        """
        Yields pinch off current for a given gate voltage
        """
        while True:
            rf_phase = yield
            yield self.phase_fcn(rf_phase, *self._sin_params)


class MockDACChannel(InstrumentChannel):
    """
    A single dummy channel implementation
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)

        self.add_parameter('voltage',
                           parameter_class=Parameter,
                           initial_value=0.,
                           label=f"Voltage_{name}",
                           unit='V',
                           vals=Numbers(-2., 2.),
                           get_cmd=None, set_cmd=None)


class MockDAC(Instrument):

    def __init__(
        self,
        name: str = 'mdac',
        num_channels: int = 10,
        **kwargs):

        """
        Create a dummy instrument that can be used for testing

        Args:
            name: name for the instrument
            gates: list of names that is used to create parameters for
                            the instrument
        """
        super().__init__(name, **kwargs)

        # make gates
        channels = ChannelList(self, "channels", MockDACChannel)
        for n in range(num_channels):
            num = str(n + 1).zfill(2)
            chan_name = f"ch{num}"
            channel = MockDACChannel(parent=self, name=chan_name)
            channels.append(channel)
            self.add_submodule(chan_name, channel)
        self.add_submodule("channels", channels)


class MockReadoutInstrument(Instrument):

    def __init__(
            self,
            name: str,
            parameter_name: str,
            setter_param: Parameter,
            parameter_class: callable = RFReadoutParameter,
            **kwargs):
        super().__init__(name=name)
        self.add_parameter(
            parameter_name,
            parameter_class=parameter_class,
            setter_param=setter_param,
            initial_value=0,
            get_cmd=None,
            set_cmd=None,
            **kwargs
        )


class MockLockin(MockReadoutInstrument):

    def __init__(
            self,
            name: str,
            **kwargs):
        super().__init__(
            name=name,
            parameter_name="X",
            parameter_class=TopgatedABRingParameter,
            **kwargs
        )
        self.add_parameter("frequency",
                           parameter_class=Parameter,
                           initial_value=125.,
                           unit='Hz',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("amplitude",
                           parameter_class=Parameter,
                           initial_value=0.,
                           unit='V',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("phase",
                           parameter_class=Parameter,
                           initial_value=0.,
                           unit='deg',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("time_constant",
                           parameter_class=Parameter,
                           initial_value=1.e-3,
                           unit='s',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("Y",
                           parameter_class=Parameter,
                           initial_value=0.,
                           unit='V',
                           get_cmd=None, set_cmd=None)


class MockRF(Instrument):

    def __init__(
            self,
            name: str,
            **kwargs):
        super().__init__(name=name, **kwargs)
        self.add_parameter("frequency",
                           parameter_class=Parameter,
                           initial_value=530e6,
                           unit='Hz',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("power",
                           parameter_class=Parameter,
                           initial_value=7.,
                           unit='dBm',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("phase",
                           parameter_class=Parameter,
                           initial_value=0.,
                           unit='deg',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("status",
                           parameter_class=Parameter,
                           initial_value=False,
                           get_cmd=None, set_cmd=None)


class MockField(Instrument):

    def __init__(
            self,
            name: str,
            vals: Numbers = Numbers(min_value=-1., max_value=1.),
            start_field: float = 0.0,
            **kwargs):
        super().__init__(name=name, **kwargs)
        self._field = start_field
        self.add_parameter("field",
                           parameter_class=Parameter,
                           initial_value=start_field,
                           unit='T',
                           vals=vals,
                           get_cmd=self.get_field, set_cmd=self.set_field)
        self.add_parameter("ramp_rate",
                           parameter_class=Parameter,
                           initial_value=0.1,
                           unit='T/min',
                           get_cmd=None, set_cmd=None)
        self._ramp_start_time = None
        self._wait_time = None
        self._fr = self._field_ramp()
        next(self._fr)

    def get_field(self):
        """
        This method is automatically wrapped to
        provide a ``get`` method on the parameter instance.
        """
        if self._ramp_start_time:
            _time_since_start = time.time() - self._ramp_start_time
            val = self._fr.send(_time_since_start)
            next(self._fr)
            self._field = val
        return self._field

    def set_field(self, value, block: bool = True):
        if self._field == value:
            return value

        self._wait_time = 60. * np.abs(self._field - value) / self.ramp_rate()
        self._field_ramp_fcn = interp1d(
            x=[0.0, self._wait_time],
            y=[self._field, value],
            bounds_error=False,
            fill_value=(self._field, value)
        )
        self._ramp_start_time = time.time()

        if block:
            time.sleep(self._wait_time)
            self._field = value
            return value

    def _field_ramp(self):
        """
        Yields pinch off current for a given gate voltage
        """
        while True:
            _time = yield
            if _time is None:
                _time = 0.0

            yield float(self._field_ramp_fcn(_time))
