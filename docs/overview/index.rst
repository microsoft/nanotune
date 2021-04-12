
============
What it does
============

Nanotune was implemented as part of a PhD project initializing gate-defined quantum dots without human interaction and no pre-measured input. It automates typical manual measurements and replaces the experimenter's quality and charge state assessment by supervised machine learning.
Hard-coding the tuning logic, the implemented dot-tuning sequence can be thought of, in a very simplified way, as a for-loop with a couple of if-statements.
It only requires
Bonding scheme/routing (i.e. which DAC channel corresponds to which gate),
device layout (i.e. number and arrangement of gates),
safety voltage ranges (i.e. which voltages can be applied without damaging the device) ,
setup specific noise floor (to ensure noise is not mistaken for an open regime in pre-calibration steps).

.. _fig_gen:
.. figure:: workflow_small.png
    :alt: nanotune
    :align: center
    :width: 60.0%

The dot-tuning sequence was demonstrated on double quantum dots formed in a GaAs two-dimensional electron gas, however no assumptions about the material were made. The gate layout of the devices used, shown on the right, consist of six gates: four so-called barriers and two plungers.
Barrier gates typically create potential wells, defining tunnel resistances between dots and reservoirs, while plungers are used to adjust the electrochemical potential of a dot, changing the number of electrons if desired.
Except for the top barrier, the tuning is identical to the tuning of quantum dots formed in nanowires. All measurements were taken in DC transport, i.e. measuring current through the device, but can be replaced by sensing techniques without difficulties.

The dot-initialization workflow consists of two pre-calibration, one characterization and one tuning step, as outlined in the flowchart on the right. A gate leakage test and initial quality assessment ensure that devices are 'alive', meaning that voltages can be set and the current through the device is above the specified noise floor. The characterization step identifies devices that are fully functional, which is defined as all gates being able to deplete the electron gas nearby, also referred to as pinching off.
Similar to a manual approach, each gate is characterized individually by stepping over its safety range while measuring the current through the device. The desired feature, a sharp current dip reaching zero, is confirmed by a binary classifier trained with experimental data. Only if traces of all gates show a good pinch-off, a device is tuned.

The subsequent dot-tuning sequence is able to tune into either the single and double dot regime, with the only difference being the voltage value set to the central barrier. To form a double dot, a more negative value is required than for the single dot regime.
Specifically, the dot-tuning process consists of a sequence of one- and two-dimensional measurements, referred to as gate characterizations and charge diagrams respectively. Each measurement is assessed by a binary classifier to determine quality and, in the case of a charge diagram, the charge state (i.e. single vs double dot).
Gate characterizations are used to determine each gate's 'active' voltage range. An active voltage range, also called current_valid_range, is the range within which we expect the desired charge state to occur. A sequence of faster, one-dimensional measurements is an efficient way to narrow down the large parameter space before proceeding with more time consuming, two-dimensional measurements. Due to capacitive coupling, active ranges need to be updated each time the voltage of a nearby gate is changed.
Once all gates are characterized and barriers set to a value within their active range, the plungers are swept to measure a charge diagram. Note that the top barrier is set first and that the remaining tuning sequence is the same as for nanowires.
Depending on the classification outcome of the charge diagram as well as additional checks assessing whether the device is in an open, closed or intermediate transport regime, gate voltages are adjusted. The precise sequence of measurements is shown in the second flow diagram on the right.


The main modules implementing the dot-tuning are the Device, DeviceTuner, TuningStage, DataFit and Classifier classes. As illustrated in the diagram below, these classes work together as follows:
A subclass of a TuningStage, currently either GateCharacterization1D or ChargeDiagram, is responsible for taking data and verifying whether the result is satisfying. To do the latter, a fit is performed via one of the DataFit subclasses and either the extracted feature vector or the entire measurement is classified. GateCharacterization1D uses a PinchoffFit, while ChargeDiagram relies on DotFit for fitting.
The DeviceTuner class implements the tuning sequence itself by combining instances of TuningStages with a hard-coded decision logic leading to a dot being formed. There are two DeviceTuner subclasses, a Characterizer performing the a device characterization and a DotTuner.
The DeviceTuner acts on an instance of a Device class, which represents the physical device including a list of gates, ohmics, gate layout and readout methods.

.. _fig_gen:
.. figure:: algorithm_dot_tuning.png
    :alt: A 2DEG heterostructure
    :align: center
    :width: 60.0%


Data flow
---------

Nanotune extends QCoDeS functionalities of a data acquisition software by adding  automated tuning procedures which use machine learning models implemented in scikit-learn or tensorflow.

The diagram illustrates the data and instruction flow of quantum measurements.
The dotted ellipses indicates which stages are covered by QCoDeS or nanotune. QCoDeS provides an interface to room-temperature instruments (i.e. drivers) and tools to take and save measurements. Nanotune extends this functionality by automating common procedures encountered during quantum dot initialization. Supervised machine learning models replace the experimenter's assessment of a measurement outcome.

Nanotune's TuningStage subclasses, which are responsible for data acquisition and part of a tuning sequence of DeviceTuner class, use QCoDeS'  measurement context manager to take measurements. The data and metadata is saved via qc.Dataset to a SQLite database. In the current case, measurements are linearly spaced one- and two-dimensional traces, called GateCharacterization1D and ChargeDiagram respectively.
Once measured, the data is loaded into nanotune's dataset, where it is postprocessed, e.g. normalized with constants which were previously measured and saved to metadata. If required, Fourier frequencies or filtered data is computed as well. Next, the DataFit classes fit the data to extract features (example: slope and amplitude of a pinchoff-curve) or determine the device's transport regime (i.e. open, intermediate or closed, depending on current strength). Either the extracted feature vector or entire measurement is passed to the classifier for quality or charge state prediction. Based on the outcome, a decision about subsequent tuning is made.

