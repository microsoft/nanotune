======
Device
======

An instance of a nt.Device class represents the physical device/sample to be tuned. Rather than being a collection of instruments and instrument parameters, it should be thought of as a container keeping track of the dot-initialization process. Beside lists of gates and ohmics, a nt.Device contains information about how it should be read out (readout_methods dictionary) and how gates are arranged (device_type string). For the hard-coded tuning logic approach, knowledge about the gate layout is required and specified as a list of gate names, sorted by the order they appear on the sample. Supported device layouts are defined in the configuration file, e.g. "doubledot_2D":["top_barrier", "left_barrier", "left_plunger", "central_barrier", "right_plunger", "right_barrier"].
The class also has a convenience methods manipulating all gates collectively, which are used for initialization, resetting and normalization constant measurements. Normalization constants, a nt.Device parameter, need to be measured before a measurement by recording the noise floor and open device signal, e.g. max current when all gates are at their highest allowed voltages. They are required for data normalization before fitting and classification.

A nt.Gate is a wrapper around a voltage parameter of a DAC channel with added tuning functionalities. It contains attributes keeping track in which range each gate is expected to be 'active' (current_valid_range), at which voltage it is expected to pinch off (transition_voltage), the maximum voltage jump supported (max_jump), to be used to calculate safe setpoints if not ramping), and whether it should be ramped and ramping is supported by hardware. New voltages are ramped to if specified by the use_ramp parameter, either using a hardware ramp or QCoDeS' software ramps specified by inter_delay.
The layout_id parameter is the gate's index in the device layout list.

.. automodule:: nanotune.device
    :members:

.. toctree::
   :maxdepth: 3
