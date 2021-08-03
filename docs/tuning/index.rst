.. _tuning:

Tuning
======

The :ref:`device_tuner` module implements the tuning outlined in the workflow diagram
:numref:`fig_workflow`. Device characterization and dot tuning are covered by
two separate sub-classes of the `Tuner` base: `Characterizer` and `DotTuner`.
A tuner class acts on an instance of :ref:`device` to perform the tuning.
Each of them uses two different types of :ref:`tuningstages`, the
`GateCharacterization1D` and `ChargeDiagram`, to take and analyse data. Tuning
results are saved in an instance of a `TuningHistory`, relying on `TuningResult`
and `MeasurementHistory`, all of which are described in :ref:`results_saving`.


.. toctree::
   :maxdepth: 2

   device_tuner
   tuningstages
   results_saving
