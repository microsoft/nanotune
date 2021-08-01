======
Data
======

The nt.Dataset class loads QCoDeS data into a xarray via the to_xarray_dataset
method of the qc.Dataset.
The raw data, stored in the raw_data attribute, is normalized by normalization
constants loaded from metadata, currently stored in a separate column in the
database called "nanotune_metadata". The normalized data is stored in the data
attribute, also a xarray.Dataset. filtered_data and power_spectrum attributes
contain Gaussian filtered data and Fourier frequencies respectively. They are
computed by the prepare_filtered_data and compute_power_spectrum methods.

The abstract DataFit class is the base class for fitting classes. It's purpose
is to ensure that sub-classes have a find_fit and plot_fit method, as well as a
next_actions attribute which are called when running using an instance of a
TuningStage.
The plot_fit method extracts features and determines the transport regime
(open, closed, intermediate). In case of a closed or open regime,
the next_actions list is populated with strings suggesting how voltages need to
be adjusted. If the normalized current is below a lower threshold,
"more positive" is added and "more negative" if the normalized current is above
a upper threshold.

The PinchoffFit class implements a hyperbolic tangent fit to extract features
such as amplitude, slope, residuals etc. It also determined the active range of
the gate swept, indicated by L and H in the plot on the left, and the
transition voltage, indicated by T. Both, the active range and the transition
voltage, are calculated based in the first derivative of either the fit or
normalized data.

The DotFit class aims to locate triple points. However the current
implementation only works with excellent data and has not been used for the
autonomous tuning paper. Similarly, CoulomboscillatioFit has not been used
either.



The dot-tuning sequence takes two types of measurements: one-dimensional
pinch-off curves and two-dimensional charge diagrams. Charge diagram come in
four flavors: good single dot, poor single dot, good double dot, poor double
dot. Pinch-off curves are labelled good or poor.
The two measurements on the right show the results nanotune is looking for:
a good pinch-off, also referred to as 1D gate characterization, and a good
double dot charge diagram.

Pinch-off
---------
Examples of poor and good pinch-off curves. The label poor is attached to
measurements where the current doesn't reach zero or if the current drop is
not clear. A flatter drop is still considered good as what we care about most
is a gate's ability to deplete the electron gas nearby.

Single dot
----------
Good single dots show clear and sharp diagonal lines. Taking one-dimensional
traces give typical Coulomb oscillation sweeps.
Taking a larger scan can look like the measurement below.
nanotune avoid these large sweeps by doing a GateCharacterization1D
beforehand, determining more precise ranges for both gates.
Poor single dot. A dot starts to form but diagonal lines are not sharp. 1D
Coulomb oscillations would show broad, doubled, or any other deformed peaks.


Double dot
----------
In an ideal case, the final charge diagram of a coarse tuning algorithm shows
triple points easily locatable with a fine-tuning algorithm. In general, a
double dot regime can look different between tune-ups or devices.

Data flow
---------

The diagram illustrates the data and instruction flow of quantum measurements.
The dotted ellipses indicates which stages are covered by QCoDeS or nanotune.
QCoDeS provides an interface to room-temperature instruments (i.e. drivers)
and tools to take and save measurements. Nanotune extends this functionality
by automating common procedures encountered during quantum dot initialization.
Supervised machine learning models replace the experimenter's assessment of a
measurement outcome.

Nanotune's TuningStage subclasses, which are responsible for data acquisition
and part of a tuning sequence of DeviceTuner class, use QCoDeS'  measurement
context manager to take measurements. The data and metadata is saved via
qc.Dataset to a SQLite database. In the current case, measurements are
linearly spaced one- and two-dimensional traces, called
GateCharacterization1D and ChargeDiagram respectively.
Once measured, the data is loaded into nanotune's dataset, where it is
postprocessed, e.g. normalized with constants which were previously
measured and saved to metadata. If required, Fourier frequencies or
filtered data is computed as well. Next, the DataFit classes fit the
data to extract features (example: slope and amplitude of a pinchoff-curve)
or determine the device's transport regime (i.e. open, intermediate or
closed, depending on current strength). Either the extracted feature
vector or entire measurement is passed to the classifier for quality or
charge state prediction. Based on the outcome, a decision about
subsequent tuning is made.

