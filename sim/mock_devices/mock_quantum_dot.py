""" Defines mock quantum dot devices"""

# pylint: disable=too-many-arguments, too-many-locals
from typing import Mapping, Sequence

from .mock_device import MockDevice
from .mock_pin import MockPin
from sim.mock_pin import IMockPin

# pylint: enable=too-few-public-methods

class MockSingleQuantumDot(MockDevice):
    """ Mock device for a 3 gate single quantum dot """
    def __init__(self, name: str):
        super().__init__(
            name,
            [
                MockPin("src"),
                MockPin("drain"),
                MockPin("left_barrier"),
                MockPin("right_barrier"),
                MockPin("plunger"),
            ]
        )

    @property
    def src(self) -> IMockPin:
        """ Source Pin """
        return self["src"]

    @property
    def drain(self) -> IMockPin:
        """ Drain Pin """
        return self["drain"]

    @property
    def left_barrier(self) -> IMockPin:
        """ Left Barrier Gate Pin """
        return self["left_barrier"]

    @property
    def right_barrier(self) -> IMockPin:
        """ Right Barrier Gate Pin """
        return self["right_barrier"]

    @property
    def plunger(self) -> IMockPin:
        """ Central Plunger Gate Pin """
        return self["plunger"]


class MockDoubleQuantumDot(MockDevice):
    """Represents a mock quantum dot device"""

    def __init__(self, name: str):
        super().__init__(
            name,
            [
                MockPin("src"),
                MockPin("top_barrier"),
                MockPin("left_barrier"),
                MockPin("left_plunger"),
                MockPin("central_barrier"),
                MockPin("right_plunger"),
                MockPin("right_barrier"),
                MockPin("drain"),
            ],
        )

    @property
    def src(self) -> IMockPin:
        """Source Pin"""

        return self["src"]

    @property
    def top_barrier(self) -> IMockPin:
        """Top Barrier Pin"""

        return self["top_barrier"]

    @property
    def left_barrier(self) -> IMockPin:
        """Left Barrier Pin"""

        return self["left_barrier"]

    @property
    def left_plunger(self) -> IMockPin:
        """Left Plunger Pin"""

        return self["left_plunger"]

    @property
    def central_barrier(self) -> IMockPin:
        """Central Barrier Pin"""

        return self["central_barrier"]

    @property
    def right_plunger(self) -> IMockPin:
        """Right Plunger Pin"""

        return self["right_plunger"]

    @property
    def right_barrier(self) -> IMockPin:
        """Right Barrier Pin"""

        return self["right_barrier"]

    @property
    def drain(self) -> IMockPin:
        """Drain pin. This is the output device of the quantum dot."""

        return self["drain"]
