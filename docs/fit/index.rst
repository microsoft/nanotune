.. _datafitting:

Data fitting
============

Date fitting base class
-----------------------

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

.. toctree::
   :maxdepth: 2

   pinchofffit
   dotfit
   coulomboscillationfit
