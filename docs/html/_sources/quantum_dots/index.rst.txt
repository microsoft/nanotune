.. title:: Quantum Dots


Quantum Dots
===============

Quantum dots are artificially structured systems confining charges (in our case electrons) on a small scale such that electron-electron interactions such as Coulomb interactions play a role. In fact, the Coulomb blockade effect is one of the fundamental transport phenomena here.

Coulomb blockade is a classical effect arising due to the Coulomb repulsion of electrons, preventing them to flow through the quantum dot. Thus the current-voltage relation no longer follows Ohm's law but looks like a staircase instead.

The quantum dots we care about here are defined in either a semiconductor heterostructure or nanowire, hosting an electron gas. To form a dot,  voltages are applied to gate electrodes located at proximity of the semiconductor.


Dots in a two-dimensional heterostructure (2DEG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _fig_gen:
.. figure:: quantum_dots-18.svg
   :alt: A 2DEG heterostructure
   :align: center
   :width: 50.0%

   A 2DEG heterostructure.

Mobile electrons from a thin silicon donor layer within  AIGaAs layer are attracted by GaAs - AIGaAs  interface, marked in red on the left. This is due to the lower energy (energy profile depicted in black) they carry there. This effectively confines charges on a 2D plane.

By applying voltages to electrostatic gates at the surface (black picture on top, with gold-colored electrodes), a potential profile with barriers and wells is created. This confines electrons in remaining dimensions (1D, 0D). 


.. _fig_gen:
.. figure:: quantum_dots-22.svg
   :alt: A 2DEG heterostructure
   :align: center
   :width: 50.0%

   A 2DEG heterostructure.

Once formed, quantum dots can be modelled as conducting islands, shown in grey, connected to source (S) and drain (D), shown in blue. Couplings between dots and reservoirs are modelled as a resistor in parallel to a capacitor.


One way to determine a quantum dot's properties is to measure current through it, i.e. perform a DC transport measurement. In such a measurement, a small bias between source and drain is applied. If the dots' energy levels are within the small bias window, then electron transport will occur. One often omits the small bias and talks about the alignment of the dots energy levels, as shown in the left figure below.
If the levels don't align, as pictured on the right, the dot is in the Coulomb blockade regime: Coulomb repulsion between electrons prevents multiple electrons to occupy the same energy level and transport is suppressed.

.. _fig_gen:
.. figure:: quantum_dots-19.svg
   :alt: A 2DEG heterostructure
   :align: center
   :width: 50.0%

   A 2DEG heterostructure.

.. _fig_gen:
.. figure:: quantum_dots-20.svg
   :alt: A 2DEG heterostructure
   :align: center
   :width: 50.0%

   A 2DEG heterostructure.


.. _fig_gen:
.. figure:: quantum_dots-07.svg
    :alt: A 2DEG heterostructure
    :align: center
    :width: 50.0%

    Electrons tunnel individually as the energy required to add two electrons is significantly higher than to add one.


.. _fig_gen:
.. figure:: quantum_dots-06.svg
    :alt: A 2DEG heterostructure
    :align: center
    :width: 50.0%

    Coulomb blockade regime: no transport occurs/no current is measured. The green electron has neither the required energy to occupy the upper energy state εN+1, nor can it occupy εN due to Coulomb repulsion with the electron there.
If one energy level is within the bias window, a measurable current arises due to a so-called co-tunneling processes via virtual states.


Transport features of dots with well-localized and weakly coupled charges can be explained and qualitatively reproduced by the classical capacitance model, which represents gates, dots and reservoirs as conductors connected through resistors and capacitors. It also allows to capture the so-called gate cross-talk, i.e. the effect of capacitive couplings of all gates to each dot.

.. _fig_gen:
.. figure:: quantum_dots-08.svg
    :alt: A 2DEG heterostructure
    :align: center
    :width: 50.0%

    Capacitive coupling between gates and gates and dots of a double dot device with six electrostatic gates.


.. _fig_gen:
.. figure:: quantum_dots-09.svg
    :alt: A 2DEG heterostructure
    :align: center
    :width: 50.0%

    Schema of a double dot device with six electrostatic gates.