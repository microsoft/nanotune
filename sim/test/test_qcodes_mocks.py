from sim.mock_devices import MockDevice
from sim.qcodes_mocks import MockDeviceInstrument, SimulationParameter
from sim.mock_devices import Pin

def test_simulation_parameter_with_init():
    """Verifies that a SimulationParameter instance can be properly initialized
       by QCoDeS when adding a parameter to an instrument with a default initial_value"""

    class TestDevice(MockDevice):
        def __init__(self):
            super().__init__(
                "mock_device",
                [
                    Pin("pin")
                ]
            )

        @property
        def pin(self):
            return self["pin"]


    class TestInstrument(MockDeviceInstrument):
        def __init__(self):
            super().__init__("mock_instr", TestDevice())

            self.add_parameter(
                "src",
                parameter_class=SimulationParameter,
                unit="V",
                sim_pin=self.mock_device.pin,
                initial_value = 1.23
            )

    test_inst = TestInstrument()
    assert test_inst.src() == 1.23, "The Test Instrument failed to initialize the src parameter value"

