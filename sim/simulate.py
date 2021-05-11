#pylint: too-many-arguments, too-many-locals
from typing import Union
from sim.data_providers import StaticDataProvider, QcodesDataProvider, Pin


class InputPin:
    """Represents an input pin on a device.  Can be bound to a data provider as
    an input source.
    """

    def __init__(
        self,
        name: str,
        default_value: float = 0.0,
    ) -> None:
        self._name = name
        self._value = default_value

    def __repr__(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        """Name of the input pin"""

        return self._name

    @property
    def value(self) -> float:
        """Gets the current value of the pin"""

        return self._value

    def get_value(self) -> float:
        """Gets the current value on the input pin.  Compatible with qcodes
        Parameter get_cmd argument.
        """

        return self._value

    def set_value(self, value: float) -> None:
        """Sets the current value on the input pin.  Compatible with qcodes
        Parameter set_cmd argument.
        """

        self._value = value



class OutputPin:
    """Represents an output pin on a device.  Can be associated with a data
    provider to generate ouput data given bound inputs.
    """

    def __init__(
        self,
        name: str,
        default_data_provider: StaticDataProvider = StaticDataProvider(0.0),
    ) -> None:
        """Initializes an OutputPin."""

        self._name = name
        self._data_provider = default_data_provider

    def __repr__(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    def set_data_provider(
        self,
        data_provider: StaticDataProvider,
    ) -> None:
        """Sets the current data provider for the output pin's values"""

        self._data_provider = data_provider

    @property
    def name(self) -> str:
        """Name of this output pin"""

        return self._name

    @property
    def value(self) -> float:
        """Returns the current value of this output pin, as determined by the
        current data provider.
        """

        return self._data_provider.value

    def get_value(self) -> float:
        """Returns the current value of this output pin. Compatible with
        qcodes Parmeter get_cmd argument.
        """

        return self._data_provider.value


#pylint: disable=too-few-public-methods
class SimulatedDevice:
    """Base class for simulated devices """

    def __init__(self, pins: List[str]):
        self._pins = {pin.name:pin for pin in pins}

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        input_pins = [
            str(pn) for pn in self._pins.values() if isinstance(pn, InputPin)
        ]
        inputs  = "Inputs  : {0}".format(", ".join(input_pins))
        output_pins = [
            str(pn) for pn in self._pins.values() if isinstance(pn, OutputPin)
        ]
        outputs = "Outputs : {0}".format(", ".join(output_pins))
        return "\n".join([inputs, outputs])

    def __getitem__(self, pin_name: str) -> Union[InputPin, OutputPin]:
        return self._pins[pin_name]
#pylint: enable=too-few-public-methods


class QuantumDotSim(SimulatedDevice):
    """Represents all gates on a quantum dot."""

    def __init__(self):
        super().__init__([
            InputPin("src"),
            InputPin("l_barrier"),
            InputPin("l_plunger"),
            InputPin("c_barrier"),
            InputPin("r_plunger"),
            InputPin("r_barrier"),
            OutputPin("drain") ])

    @property
    def src(self) -> str:
        """Source Pin"""

        return self["src"]

    @property
    def l_barrier(self) -> InputPin:
        """Left Barrier Pin"""

        return self["l_barrier"]

    @property
    def l_plunger(self) -> InputPin:
        """Left Plunger Pin"""

        return self["l_plunger"]

    @property
    def c_barrier(self) -> InputPin:
        """Central Barrier Pin"""

        return self["c_barrier"]

    @property
    def r_plunger(self) -> InputPin:
        """Right Plunger Pin"""

        return self["r_plunger"]

    @property
    def r_barrier(self) -> InputPin:
        """Right Barrier Pin"""

        return self["r_barrier"]

    @property
    def drain(self)-> OutputPin:
        """Drain pin. This is the output device of the quantum dot."""

        return self["drain"]
