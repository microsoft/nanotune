.. _tuningstages:

Tuningstages
============

The TuningStage abstraction represents a typical step of both manual and
automated tuning, its workflow is illustrates on the left. The TuningStage
class itself is meant to be sub-classed, where the sub-classes implement the
bodies of some of the methods part of the workflow.

A TuningStage takes as input a list of gates to be swept as well as the
(trained) machine learning models to use for classification. Setpoints are
calculated based on the gates' current_valid_ranges, voltages ranges
previously determined to be promising. The current method returns linearly
spaced setpoints but is meant to be changed in the future.

Once the data is taken, it is fitted via a DataFit sub-class. Either the
extracted feature vector or the entire measurement (normalized, filtered
or Fourier frequencies) are passed to the classifier for a quality or regime
assessment.
If the desired quality/regime is reached, a TunigResult instance is returned.
If not, the current strength is checked to determine whether the current gate
voltage ranges need to be adjusted. If the detected current is too low/high,
voltage ranges are adjusted towards more positive/negative values.
If the new ranges do not exceed the gates' safety ranges, the measurement
loop resumes by calculating new setpoints. If the gates' ranges cannot be
changes, the TuningStage returns an instance of a TuningStage indicating a
non-successful stage.

The TuningStage abstract base class contains all necessary attributes and
methods to implement the workflow above. It's main method is run_stage,
which makes use of a few abstract methods, such as check_quality,
update_current_ranges and get_next_actions. These depend on the type of
measurements and hence require specific implementations. Currently implemented
subclasses are GateCharacterization1D and ChargeDiagram. If required, these
have additional attributes or methods.


Gate characterization 1D
------------------------

Charge diagram
--------------
