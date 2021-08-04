.. _device_tuner:

Device tuner
============


Tuner
-----

The base class implements the most common tuning methods such as updating a
device's normalization constants, measuring a pinchoff or taking a charge diagram.
It also has a method to determine the initial ranges of a helper gate, such as
a top barrier of a 2D layout or bottom gate underneath the entire 1D system.

Measurement settings
    Measurement related settings are grouped in two dataclasses, `DataSettings`
    and `SetpointSettings`. A tuner class keeps an instance of each with
    general settings used for tuning of all devices. Examples of setpoint
    setting used
    for all measurements are setpoint_method, optionally voltage precision and
    high resolution precisions. As for data settings, data base location or
    size of charge diagram segments used for classification also stay unchanged
    in most cases.
    `Tuner` provides methods to compile device specific settings, by merging
    default with device specific settings.

The tuner classes `Characterizer` and `DotTuner` inherit from this base class.


Characterizer
-------------

The `Characterizer` implements the device characterization step via its
`characterize` method.

Dot tuner
---------

The `DotTuner` has all methods to implement the tuning procedure shown in figure
:numref:`tuning_algorithm`.
The entire sequence is implemented in the `tune` method, however all methods can
be used independently to realize similar but different sequences.

.. _tuning_algorithm:
.. figure:: ../overview/algorithm_dot_tuning.svg
   :alt: dot tuning algorithm
   :align: center
   :width: 55 %

   Dot tuning algorithm implemented by the `DotTuner` class.

The steps outlined in the diagram have a corresponding method within the class.

+--------------------------------------+---------------------------------------+
|step                                  |                 method                |
+--------------------------------------+---------------------------------------+
|Set top barrier                       |           `set_helper_gate`           |
+--------------------------------------+---------------------------------------+
|Characterize central barrier          |                                       |
|Set central barrier                   |          `set_central_barrier`        |
+--------------------------------------+---------------------------------------+
|Characterize outer barriers           |                                       |
|Set outer barriers                    |          `set_outer_barriers`         |
+--------------------------------------+---------------------------------------+
|Characterize plungers                 |         `characterize_plunger`        |
+--------------------------------------+                                       |
|                                      |  (used in `set_valid_plunger_ranges`) |
+--------------------------------------+---------------------------------------+
|Characterize charge stability diagram |          `get_charge_diagram`         |
|Classify charge state                 |                                       |
+--------------------------------------+---------------------------------------+
|`get_charge_diagram` returns classification result in TuningResult instance   |
|and changes plunger ranges if needed,                                         |
|executing the small inner lop on the left.                                    |
+--------------------------------------+---------------------------------------+
|Change central barrier                |      `update_gate_configuration`      |
+--------------------------------------+---------------------------------------+


The loop going left and right from "Classify state" are implemented by
`update_gate_configuration`, which calls `adjust_all_barriers_loop` and which
in turn calls `adjust_all_barriers`. This sets the top and central barrier.

The large left loop:
 - uses termination reasons to update outer barriers first (if device was too pinched off or open)
 - update top barrier/helper gate in `adjust_all_barriers`, called in `adjust_all_barriers_loop`
 - update top barrier

The right loop:
    - change central barrier with `initial_voltage_update` in `adjust_all_barriers_loop`.
    - if central barrier is set successfully, the loop in `adjust_all_barriers_loop` is not executed and thus tuning resumes with plunger range characterization.
    - if after setting the central barrier other changes are needed, the loop in `adjust_all_barriers_loop` will set the top barrier, then central and also outer barriers.

In both loops: of outer barrier can not be set, the top barrier is changed again.

`set_central_and_outer_barriers` first sets the central and outer barriers beforehand
it loops:
1. update helper gate
2. set outer barriers
until suitable values are found.

`adjust_all_barriers_loop`: update voltages based on input directives and run `adjust_all_barriers` in a loop.

`choose_new_gate_voltage`:
