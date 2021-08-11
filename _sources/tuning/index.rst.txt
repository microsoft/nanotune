.. _tuning:

Tuning
======

Tuning quantum dots is a process roughly consisting of two parts. First, puddles
of electrons are formed within an electron gas by finding appropriate
voltages for all electrostatic gates nearby. The second step optimizes the
voltages of a small subset of these gates to adjust the number of charges within
each dot and the couplings between them. The figure below illustrates the entire
tuning process on a nanowire, where both steps have been performed.


.. |nw13| image:: ./figs/nw_dots-13.svg
   :width: 45 %

.. |labels| image:: ./figs/nw_dots-08.svg
   :width: 15 %

.. |array| image:: ./figs/nw_dots-20.svg
   :width: 8 %

.. |nw09| image:: ./figs/nw_dots-09.svg
   :width: 40 %

|nw13| |array| |nw09|

|labels|

The general goal of dot tuning illustrated on a nanowire: by setting voltages to
all electrostatic gates, well-defined quantum dots with specific tunnel
couplings can be formed.

A manual approach to tuning consists of a sequence of measurements iteratively
narrowing down the voltage range of each gate, outlined in
:ref:`manual_tuning`. nanotune applies this approach to the first step,
also referred to as coarse tuning. By automating some of the most common
measurements and replacing the experimenter's decision making by binary
classifiers, it implements the device characterization and dot-tuning
procedures outlined in the workflow diagram in :numref:`fig_workflow`.

Both device characterization and dot tuning is covered by the tuning module,
whose main components are the `Tuner` base class and its two subclasses
`Characterizer` and `DotTuner`, described in :ref:`device_tuner`. Tuning
results are saved in an instance of a `TuningHistory`, which holds instances
of `MeasurementHistory` and `TuningResult`, described in :ref:`results_saving`.


.. toctree::
   :maxdepth: 2

   manual_tuning
   device_tuner
   tuningstages
   results_saving
