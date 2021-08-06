.. _device:

Device abstraction
==================

The physical device or sample is represented by an instance of a `Device` class, which
in turn is a QCodeS `DelegateInstrument`. It extends its base class by adding
attributes and methods specific to the characterization and
dot-initialization process.
Beside the initialization of instrument parameters and channels, the following
attributes are meant to assist in the tuning process

list of gates and ohmics
    Gates and ohmics are grouped in separate lists in the order specified by
    the device layout. The index of each gate or ohmic within the respective
    list should be the items ID, either `gate_id` or `ohmic_id`. This layout
    is specified via a subclass of `DeviceLayout`.

readout
    How and with with parameters the device should be read out. It is an
    instance of a `Readout` dataclass,
    where each
    attribute, one for each readout methods, is either a QcoDes `Parameter` or
    `GroupedParameter`. Typical readout method types are transport, sensing and rf.

main readout method
    Indicating which readout method should be used to make tuning decisions. Features
    extracted during a data fit can vary between readout methods, some of which
    can have more noise than others. nanotune's fitting procedures perform a fit
    on all of them, but only one is used decide e.g. whether the result is a
    good one or how gates should be adjusted next.
    It is added to static metadata and thus to every dataset measured.

normalization constants
    Keeping track of the highest and lowest signals, i.e. noise floor and open device
    signal. These values are required for correct classification and needs to
    be measured
    before tuning and every time settings on
    readout instruments are changed. Each readout method has its own constants.
    They are saved to static metadata of the device and thus to metadata of each
    dataset measured. A separate dataclass described below is used as a
    container.

current valid ranges
    The voltages ranges within which we expect the desired features to occur,
    determined during tuning. If known, these are the current ranges within
    which to tune. They typically depend on
    voltages of other gates and need to be updated during tuning. These
    are also the ranges which will be swept during measurements.

initial valid ranges
    The voltages ranges within which we expect the desired features to occur
    before any tuning has been done. They can be used if ranges can be narrowed down
    based on prior knowledge such as measurements of other samples or when
    a device is loaded pre-tuned.
    Will be set to each gate's safety range if not specified.

transition voltages
    Voltages at which a gate is able to deplete the electron gas nearby.
    Specifically, it is the voltage at which the measured signal drops from open
    to closed regime. These are determined during tuning and depend on voltages
    set to other gates.

quality
    Indicating whether the device is fully functional and can thus be considered
    for dot tuning.


The class also has a convenience methods manipulating all gates collectively,
which are used for initialization, resetting and normalization constant
measurements.


The device can be initialized using a yaml file. Example:

.. code-block:: yaml

    instruments:

    device_example:
      type: nanotune.device.device.Device
        init:
        device_type:
          doubledot
        normalization_constants:
          transport: [0, 2]
          sensing: [-0.3, 0.6]
        channels:
          type: nanotune.device.device_channel.DeviceChannel
          top_barrier:
            channel: dac.ch01
            gate_id: 0
          left_plunger:
            channel: dac.ch02
            gate_id: 1
          left_ohmic:
            channel: dac.ch10
            ohmic_id: 0
        readout:
          transport:
            lockin.X
          sensing:
            rf.phase
        initial_valid_ranges:
            top_barrier: [-3, 0]
        current_valid_ranges:
            top_barrier: [-0.5, 0]
        transition_voltages:
            top_barrier: -0.4

Normalization constants
-----------------------

This dataclass is used to hold and update normalization constants of a device.
Each attribute keeps track of the constants of one readout method, e.g. transport,
sensing and rf.

Device channel
--------------

A `DeviceChannel` is a wrapper of an instrument channel parameter such as a
DAC channel with added functionalities. The channel wrapped and passed to init
function needs to either implement the methods of a `DACChannelInterface` described in
:ref:`drivers`, or be in subclass thereof. This interface setup ensures that
channels of different instruments can be used, where it is up to the user to
implement the 'glue' between hardware and software.

Main attributes of `DeviceChannel` are:

gate ID
    Identifier indicating which gate of a device layout the channel
    represents. Example: a left barrier's ID of a `DoubleDotLayout` is 1.

ohmic ID
    Identifier indicating which ohmic of a sample or device layout the channel
    represents.

safety voltage range
    Voltage range within within which the gate is guaranteed not to damage the
    device.

supports hardware ramp
    Wether the instrument channel and thus instrument itself can sweep/set voltages
    with a hardware ramp.

use ramp
    Whether voltages should be set or ramped.

ramp rate
    Rate at which voltages should be ramped if `use_ramp = True`.

max voltage step
    Maximum voltage step supported by the gate, i.e. the largest voltage change
    that can be set without ramping such that the device is not damaged.

relay state
    If the DAC has relay states, this attribute indicates the current
    setting of it. Examples: ground or floating.


Device layout
-------------

The `DeviceLayout` class serves as interface for specifying a device
layout. A subclass needs to implement the methods which return the gate IDs of
device channels serving the specific purpose. For example, the method
`barriers` needs to return the gate IDs of all barriers of the device.
`DeviceLayout` is a dataclass inheriting from the abstract `IDeviceLayout`.
An example of a device layout implement is `DoubleDotLayout`.
