#pylint: disable=too-many-arguments, too-many-locals
from typing import cast
from qcodes import Instrument, Parameter

from sim.data_providers import IDataProvider
from sim.mock_device import IMockDevice
from sim.mock_devices import MockSingleQuantumDot, MockDoubleQuantumDot
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
        # must set our pin first, because super() may call set_raw on us
        self._pin = sim_pin
        super().__init__(**kwargs)

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


class MockSingleQuantumDotInstrument(MockDeviceInstrument):
    """ QCoDeS Mock Instrument that wraps a MockSingleQuantumDot device """

    def __init__(self, name: str = "MockSingleQuantumDotInstrument"):

        super().__init__(name, MockSingleQuantumDot(name))
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
            sim_pin=mock.left_barrier,
        )
        self.add_parameter(
            "right_barrier",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.right_barrier
        )
        self.add_parameter(
            "plunger",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.plunger,
        )
        self.add_parameter(
            "drain",
            parameter_class=SimulationParameter,
            unit="I",
            sim_pin=mock.drain
        )

    @property
    def mock_device(self) -> MockSingleQuantumDot:
        """ The mock device that this instrument is wrapping """
        return cast(MockSingleQuantumDot, super().mock_device)


class MockDoubleQuantumDotInstrument(MockDeviceInstrument):
    """ QCoDeS Mock Instrument that wraps a MockDoubleQuantumDot device """

    def __init__(self, name: str = "MockDoubleQuantumDotInstrument"):

        super().__init__(name, MockDoubleQuantumDot(name))
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
            sim_pin=mock.left_barrier,
        )
        self.add_parameter(
            "right_barrier",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.right_barrier
        )
        self.add_parameter(
            "central_barrier",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.central_barrier,
        )
        self.add_parameter(
            "left_plunger",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.left_plunger,
        )
        self.add_parameter(
            "right_plunger",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=mock.right_plunger,
        )
        self.add_parameter(
            "drain",
            parameter_class=SimulationParameter,
            unit="I",
            sim_pin=mock.drain
        )

    @property
    def mock_device(self) -> MockDoubleQuantumDot:
        """ The mock device that this instrument is wrapping """
        return cast(MockDoubleQuantumDot, super().mock_device)
