#pylint: disable=too-many-arguments, too-many-locals

from qcodes import Instrument, Parameter
from sim.pin import IPin
from sim.data_provider import IDataProvider
from sim.simulate import QuantumDotSim

class SimulationParameter(Parameter):
    """ Qcodes Parameter that wraps a Simulation Pin, which in turn
        uses an IDataProvider as the backing data for the pin.
    """
    def __init__(
            self,
            sim_pin : IPin,
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
    def pin(self) -> IPin:
        """ Retrieve the simulation pin that this QCoDeS parameter is wrapping """

        return self._pin

    def set_data_provider(self, data_provider : IDataProvider) -> None:
        """ Convenience method to set the data provider on the pin backing this parameter """

        self._pin.set_data_provider(data_provider)


class QuantumDotMockInstrument(Instrument):
    """ QCoDeS Mock Instrument that wraps a QuantumDotSim device """

    def __init__(self, name: str = "QuantumDotMockInstrument"):

        super().__init__(name)

        sim = QuantumDotSim(name)
        self._simulator = sim

        self.add_parameter(
            "src",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=sim.src,
        )
        self.add_parameter(
            "left_barrier",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=sim.l_barrier,
        )
        self.add_parameter(
            "right_barrier",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=sim.r_barrier
        )
        self.add_parameter(
            "central_barrier",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=sim.c_barrier,
        )
        self.add_parameter(
            "left_plunger",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=sim.l_plunger,
        )
        self.add_parameter(
            "right_plunger",
            parameter_class=SimulationParameter,
            unit="V",
            sim_pin=sim.r_plunger,
        )
        self.add_parameter(
            "drain",
            parameter_class=SimulationParameter,
            unit="I",
            sim_pin=sim.drain
        )

    @property
    def simulator(self) -> QuantumDotSim:
        """Returns the simulator instance to which this mock device is attached
        """

        return self._simulator
