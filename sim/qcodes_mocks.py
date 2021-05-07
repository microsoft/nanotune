#pylint: disable=line-too-long, too-many-arguments, too-many-locals

''' Contains qcodes mock instruments that wrap related simulators '''

from qcodes import Instrument
from sim.simulate import QuantumDotSim

class QuantumDotMockInstrument(Instrument):

    ''' QCoDeS Mock Instrument that wraps a QuantumDotSim device '''

    def __init__(self, name : str = "QuantumDotMockInstrument"):

        super().__init__(name)

        sim = QuantumDotSim()
        self._simulator = sim

        self.add_parameter("src",               set_cmd=sim.src.set_value,          get_cmd=sim.src.get_value,          unit="V")
        self.add_parameter("left_barrier",      set_cmd=sim.l_barrier.set_value,    get_cmd=sim.l_barrier.get_value,    unit="V")
        self.add_parameter("right_barrier",     set_cmd=sim.r_barrier.set_value,    get_cmd=sim.r_barrier.get_value,    unit="V")
        self.add_parameter("central_barrier",   set_cmd=sim.c_barrier.set_value,    get_cmd=sim.c_barrier.get_value,    unit="V")
        self.add_parameter("left_plunger",      set_cmd=sim.l_plunger.set_value,    get_cmd=sim.l_plunger.get_value,    unit="V")
        self.add_parameter("right_plunger",     set_cmd=sim.r_plunger.set_value,    get_cmd=sim.r_plunger.get_value,    unit="V")
        self.add_parameter("drain",             set_cmd=None,                       get_cmd=sim.drain.get_value,        unit="I")

    @property
    def simulator(self):
        ''' Returns the simulator instance to which this mock device is attached '''
        return self._simulator
