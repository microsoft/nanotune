.. _manual_tuning:

Manual tuning
=============

*In preparation*

.. |empty| image:: ./figs/nw_dots-01.svg
   :width: 45 %

.. |barriers| image:: ./figs/nw_dots-02.svg
   :width: 45 %

.. |chargediagram| image:: ./figs/nw_dots-15.svg
   :width: 45 %

.. |chargediagrammeasurement| image:: ./figs/nw_dots-11.svg
   :width: 30 %

.. |labels| image:: ./figs/nw_dots-16.svg
   :width: 15 %

.. |labelsgates| image:: ./figs/nw_dots-07.svg
   :width: 15 %

.. |singleelectron| image:: ./figs/nw_dots-06.svg
   :width: 45 %

.. |array| image:: ./figs/nw_dots-20.svg
   :width: 08 %

.. |tunnelcoupling| image:: ./figs/nw_dots-09.svg
   :width: 45 %

.. |tunnelcoupling2| image:: ./figs/nw_dots-10.svg
   :width: 45 %


.. |singlechargediagram| image:: ./figs/nw_dots-12.svg
   :width: 30 %

.. |sweepsingle| image:: ./figs/nw_dots-14.svg
   :width: 45 %


Set barriers

    |empty| |array| |barriers|

|labelsgates|

Take charge diagram

Single dot:

    |labels| |sweepsingle| |singlechargediagram|


Good single dots show clear and sharp diagonal lines. Taking one-dimensional
traces give typical Coulomb oscillation sweeps.
Taking a larger scan can look like the measurement below.
nanotune avoid these large sweeps by doing a GateCharacterization1D
beforehand, determining more precise ranges for both gates.
Poor single dot. A dot starts to form but diagonal lines are not sharp. 1D
Coulomb oscillations would show broad, doubled, or any other deformed peaks.

.. _single_dot:
.. figure:: ./figs/dotfit_deafcafe-0200-0004-0000-0165b06bd0af.svg
    :alt: Double dot fit.
    :align: center
    :width: 60.0%

    Example of a double dot fit.

Double dot:

    |chargediagram| |chargediagrammeasurement|

A double dot regime can look different between tune-ups or devices.

Fine tuning:


Single electron regime

    |singleelectron|

Tunnel coupling fine tuning
    |tunnelcoupling| |tunnelcoupling2|
