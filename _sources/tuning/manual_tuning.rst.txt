.. _manual_tuning:

Manual tuning
=============

A typical manual tuning procedure consists of a series of 1D and 2D measurements,
which narrow down the large voltage parameter space within which the desired
(dot) features occur. Between each measurement the experimenter decides which
gates to adjust and how. the It is an iterative process easily taking a significant
amount of time.
Here we outline a typical procedure using nanowires as example. We begin with
a system in its initial state:

|init| |labelsgates| |labels|

.. |init| image:: ./figs/nw_dots-13.svg
   :width: 45 %

.. |empty| image:: ./figs/nw_dots-01.svg
   :width: 35 %

.. |barriers| image:: ./figs/nw_dots-02.svg
   :width: 35 %

.. |chargediagram| image:: ./figs/nw_dots-15.svg
   :width: 35 %

.. |chargediagrammeasurement| image:: ./figs/nw_dots-11.svg
   :width: 30 %

.. |labels| image:: ./figs/nw_dots-16.svg
   :width: 15 %

.. |labelsgates| image:: ./figs/nw_dots-07.svg
   :width: 15 %

.. |singleelectron| image:: ./figs/nw_dots-06.svg
   :width: 35 %

.. |array| image:: ./figs/nw_dots-20.svg
   :width: 08 %

.. |tunnelcoupling| image:: ./figs/nw_dots-09.svg
   :width: 40 %

.. |tunnelcoupling2| image:: ./figs/nw_dots-10.svg
   :width: 35 %


.. |singlechargediagram| image:: ./figs/nw_dots-12.svg
   :width: 25 %

.. |sweepsingle| image:: ./figs/nw_dots-14.svg
   :width: 35 %

.. |singledot_barriers| image:: ./figs/nw_dots-03.svg
   :width: 35 %

In a first step the barriers are set. To do so, either 1D or pair-wise 2D
sweeps are measured to narrow down the respective ranges.
Voltages at which a gate pinches off are typically set.
Setting only outer barriers results in a large single dot, while setting the
central barrier as well isolates two puddles:

|singledot_barriers| |barriers|

Single and double dots formed by setting barrier.


The regime is verified via 2D charge diagram. One can sweep the barriers, although
usually the plungers are used.

Single dots:

    |sweepsingle| |singlechargediagram|

    Good single dots show clear and sharp diagonal lines. Taking one-dimensional
    traces give typical Coulomb oscillation sweeps.
    The lines of poor single dots or dots which start to form are not sharp. 1D
    Coulomb oscillations would show broad, doubled, or any other deformed peaks.


Double dot:

    |chargediagram| |chargediagrammeasurement|

    A double dot regime can look different between tune-ups or devices. An
    excellent charge diagram will show the honeycomb structure above.

Next, plunger ranges are adjusted to expel any surplus of charges to reach the
single electron regime:
|singleelectron|

Finally, tunnel couplings are adjusted.

|tunnelcoupling| |tunnelcoupling2|
