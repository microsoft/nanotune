.. _coulomboscillationfit:

Coulomb oscillation fit
-----------------------

The `CoulombOscillationFit` class detects peaks within a 1D trace. To this end,
the `find_peaks` routine implemented in `scipy.signal` is used. It also
determines the voltage different between consecutive peaks, which can then be
used to determine other quantities such as capacitances.

An example of a Coulomb oscillation fit is shown below. Note that this routine
is currently to very robust against noise, i.e. some noises are mistaken for peaks.
This fitting procedure has not been used in the demonstration of the initial
nanotune tuning procedure (autonomous tuning paper).

.. _coulomb_fit:
.. figure:: ./figs/coulomboscillationfit_deafcafe-0200-0004-0000-01659f4b7467.svg
    :alt: Coulomb oscillation fit.
    :align: center
    :width: 60.0%

    Example of a Coulomb oscillation fit.
