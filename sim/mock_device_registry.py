from typing import Dict, Optional
from sim.mock_device import IMockDevice
from sim.mock_pin import IMockPin


class MockDeviceRegistry:

    """Acts as a global registry for all mock device instances
       so they can be easily accessed by name from any module
    """

    _mock_devices: Dict[str, IMockDevice] = {}

    @classmethod
    def register(
        cls, mock_device: IMockDevice, name: Optional[str] = None
    ) -> None:

        """Register an IMockDevice using the mock device's name, or an optionally provided name"""

        mock_name = mock_device.name if name is None else name
        cls._mock_devices[mock_name] = mock_device

    @classmethod
    def get(cls, name: str) -> IMockDevice:

        """Retrieve the mock device registered under the given name"""

        return cls._mock_devices[name]

    @classmethod
    def resolve_pin(cls, full_pin_name: str) -> IMockPin:

        """Returns the pin of a mock device given full pin name
        e.g.  "qdsim.drain" would be the name of the "drain" pin on the "qdsim" mock device"""

        device_name, pin_name = full_pin_name.split(".")
        return cls.get(device_name).get_pin(pin_name)
