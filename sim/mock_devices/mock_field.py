""" Defines mock field devices"""

# pylint: disable=too-many-arguments, too-many-locals

from .mock_device import MockDevice
from .mock_pin import MockPin

from sim.data_providers.ramped_value_data_provider import RampedValueDataProvider
from sim.mock_pin import IMockPin

class MockFieldWithRamp(MockDevice):
    """ Mock magnetic field """
    def __init__(
        self, name: str,
        ramp_rate : float = 0.1,  # T/min
        blocking : bool  = False,
        starting_value : float = 0.0
    ):
        super().__init__(
            name,
            [
                MockPin("field"),
                MockPin("ramp_rate"),
                MockPin("block")
            ]
        )

        self.ramp_rate.set_value(ramp_rate)
        self.block.set_value(1.0 if blocking else 0.0)
        self.field.set_data_provider(RampedValueDataProvider(self.ramp_rate, self.block, starting_value = starting_value))


    @property
    def field(self) -> IMockPin:
        """ Holds the value of the magnetic field """
        return self["field"]

    @property
    def ramp_rate(self) -> IMockPin:
        """ Allows the ramp_rate to be specified when changing the field """
        return self["ramp_rate"]

    @property
    def block(self) -> IMockPin:
        """ If non-zero, setting the field should block until the target setting is reached """
        return self["block"]
