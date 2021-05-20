import logging
from sim.simulate import QuantumDotSim
from sim.simulator_registry import SimulatorRegistry


class TestSimulatorRegistry:
    def test_default_registration(self):
        """Validates that a simulator auto-registers with the Simulation Registry"""

        qdsim = QuantumDotSim("sim1")

        sim1 = SimulatorRegistry.get("sim1")
        assert sim1 == qdsim

    def test_registration_with_custom_name(self):
        """Validates that a simulator auto-registers with the Simulation Registry"""

        qdsim = QuantumDotSim("sim1")
        SimulatorRegistry.register(qdsim, name="altsim")

        sim1 = SimulatorRegistry.get("sim1")
        altsim = SimulatorRegistry.get("altsim")

        assert sim1 == qdsim
        assert qdsim == altsim

    def test_resolve_pin(self):

        """Validates we can resolve a pin by name from the simulator registry"""

        qdsim = QuantumDotSim("sim1")
        drain = SimulatorRegistry.resolve_pin("sim1.drain")

        assert drain == qdsim.drain
