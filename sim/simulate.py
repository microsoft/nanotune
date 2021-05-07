#pylint: disable=line-too-long, too-many-arguments, too-many-locals

''' Contains classes related to device simulation '''

from sim.data_providers import StaticDataProvider

class InputPin:

    ''' Represents an input pin on a device.  Can be bound to a data provider as an input source '''

    def __init__(self, name, default_value = 0.0):
        self._name = name
        self._value = default_value

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

    @property
    def name(self):
        ''' Name of the input pin '''
        return self._name

    @property
    def value(self):
        ''' Gets the current value of the pin '''
        return self._value

    def get_value(self):
        ''' Gets the current value on the input pin.  Compatible with qcodes Parameter get_cmd argument '''
        return self._value

    def set_value(self, value):
        ''' Sets the current value on the input pin.  Compatible with qcodes Parameter set_cmd argument '''
        self._value = value



class OutputPin:

    ''' Represents an output pin on a device.  Can be associated with a data provider to generate ouput data given bound inputs '''

    def __init__(self, name, default_data_provider = StaticDataProvider(0.0)):
        self._name = name
        self._data_provider = default_data_provider

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

    def set_data_provider(self, data_provider):
        ''' Sets the current data provider for the output pin's values '''
        self._data_provider = data_provider

    @property
    def name(self):
        ''' Name of this output pin '''
        return self._name

    @property
    def value(self):
        ''' Returns the current value of this output pin, as determined by the current data provider '''
        return self._data_provider.value

    def get_value(self):
        ''' Returns the current value of this output pin.  Compatible with qcodes Parmeter get_cmd argument '''
        return self.value


#pylint: disable=too-few-public-methods
class SimulatedDevice:

    ''' Base class for simulated devices '''

    def __init__(self, pins : list):
        self._pins = {pin.name : pin for pin in pins}

    def __repr__(self):
        return str(self)

    def __str__(self):
        inputs  = "Inputs  : {0}".format(", ".join([str(pin) for pin in self._pins.values() if isinstance(pin, InputPin)]))
        outputs = "Outputs : {0}".format(", ".join([str(pin) for pin in self._pins.values() if isinstance(pin, OutputPin)]))
        return "\n".join([inputs, outputs])

    def __getitem__(self, pin_name):
        return self._pins[pin_name]
#pylint: enable=too-few-public-methods


class QuantumDotSim(SimulatedDevice):

    ''' Represents all gates on a quantum dot. '''

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
    def src(self):
        ''' Source Pin '''
        return self["src"]

    @property
    def l_barrier(self):
        ''' Left Barrier Pin '''
        return self["l_barrier"]

    @property
    def l_plunger(self):
        ''' Left Plunger Pin '''
        return self["l_plunger"]

    @property
    def c_barrier(self):
        ''' Central Barrier Pin '''
        return self["c_barrier"]

    @property
    def r_plunger(self):
        ''' Right Plunger Pin '''
        return self["r_plunger"]

    @property
    def r_barrier(self):
        ''' Right Barrier Pin '''
        return self["r_barrier"]

    @property
    def drain(self):
        ''' Drain pin.  This is the output device of the quantum dot '''
        return self["drain"]
