import logging

from sim.mock_device_registry import MockDeviceRegistry
from sim.mock_devices import MockQuantumDot


def test_default_registration():

    """Validates that a mock device auto-registers with the MockDeviceRegistry"""

    qdmock = MockQuantumDot("mock1")

    mock1 = MockDeviceRegistry.get("mock1")
    assert mock1 == qdmock

def test_registration_with_custom_name():

    """Validates registering a mock device with the MockDeviceRegistry using
       a custom name
       """

    qdmock = MockQuantumDot("mock1")
    MockDeviceRegistry.register(qdmock, name="altname")

    mock1 = MockDeviceRegistry.get("mock1")
    altsim = MockDeviceRegistry.get("altname")

    assert mock1 == qdmock
    assert qdmock == altsim

def test_resolve_pin():

    """Validates we can resolve a pin by name from the MockDeviceRegistry"""

    qdmock = MockQuantumDot("mock1")
    drain = MockDeviceRegistry.resolve_pin("mock1.drain")

    assert drain == qdmock.drain
