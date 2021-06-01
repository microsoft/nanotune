#pylint: disable=too-many-arguments, too-many-locals
from typing import cast
from qcodes import Instrument, Parameter

from sim.data_provider import IDataProvider
from sim.mock_device import IMockDevice
from sim.mock_devices import MockQuantumDot
from sim.mock_pin import IMockPin


class SimulationParameter(Parameter):
    """ Qcodes Parameter that wraps an IMockPin, which in turn
        uses an IDataProvider as the backing data for the value
        of the pin.
    """
    def __init__(
            self,
            sim_pin : IMockPin,
            **kwargs
         ):
        super().__init__(**kwargs)
        self._pin = sim_pin

    def get_raw(self) -> float:
        return self._pin.get_value()

    def set_raw(self, value : float) -> None:
        if self._pin.settable:
            self._pin.set_value(value)
        else:
            raise RuntimeError(
                f"Cannot set a value on pin '{self._pin.name}', "
                f"which is using a non-settable data provider")

    @property
    def pin(self) -> IMockPin:
        """ Retrieve the simulation pin that this QCoDeS parameter is wrapping """

        return self._pin

    def set_data_provider(self, data_provider : IDataProvider) -> None:
        """ Convenience method to set the data provider on the pin backing this parameter """

        self._pin.set_data_provider(data_provider)

class MockDeviceInstrument(Instrument):
    """ Base class for all qcodes mock instruments that wrap an IMockDevice """
    def __init__(self, name : str, mock_device : IMockDevice):
        super().__init__(name)
        self._mock_device = mock_device

    @property
    def mock_device(self) -> IMockDevice:
        """ The mock device that this instrument is wrapping """
        return self._mock_device


class QuantumDotMockInstrument(MockDeviceInstrument):
    """ QCoDeS Mock Instrument that wraps a QuantumDotSim device """

    def __init__(self, name: str = "QuantumDotMockInstrument"):

        super().__init__(name, MockQuantumDot(name))
        mock = self.mock_device

        self.add_parameter(
            "src",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.src,
        )
        self.add_parameter(
            "left_barrier",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.l_barrier,
        )
        self.add_parameter(
            "right_barrier",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.r_barrier
        )
        self.add_parameter(
            "central_barrier",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.c_barrier,
        )
        self.add_parameter(
            "left_plunger",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.l_plunger,
        )
        self.add_parameter(
            "right_plunger",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.r_plunger,
        )
        self.add_parameter(
            "drain",
            parameter_class=SimulationParameter,
            unit="I",
            sim_pin=mock.drain
        )

    @property
    def mock_device(self) -> MockQuantumDot:
        """ The mock device that this instrument is wrapping """
        return cast(MockQuantumDot, super().mock_device)
