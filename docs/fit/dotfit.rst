.. _dotfit:

DotFit
======

Charge diagram come in
four flavors: good single dot, poor single dot, good double dot,

The DotFit class aims to locate triple points. However the current
implementation only works with excellent data and has not been used for the
autonomous tuning paper. Similarly, CoulomboscillatioFit has not been used
either.


Single dot
----------
Good single dots show clear and sharp diagonal lines. Taking one-dimensional
traces give typical Coulomb oscillation sweeps.
Taking a larger scan can look like the measurement below.
nanotune avoid these large sweeps by doing a GateCharacterization1D
beforehand, determining more precise ranges for both gates.
Poor single dot. A dot starts to form but diagonal lines are not sharp. 1D
Coulomb oscillations would show broad, doubled, or any other deformed peaks.

.. _dot_fit:
.. figure:: ./figs/dotfit_deafcafe-0200-0004-0000-0165b06bd0af.svg
    :alt: Double dot fit.
    :align: center
    :width: 60.0%

    Example of a double dot fit.


Double dot
----------
In an ideal case, the final charge diagram of a coarse tuning algorithm shows
triple points easily locatable with a fine-tuning algorithm. In general, a
double dot regime can look different between tune-ups or devices.

.. _dot_fit:
.. figure:: ./figs/dotfit_aaaaaaaa-0000-0000-0000-016c1ca8604d.svg
    :alt: Double dot fit.
    :align: center
    :width: 60.0%

    Example of a double dot fit.

Labels
======
Charge diagram come in
four flavors: good single dot, poor single dot, good double dot, poor double
dot.
