.. title:: Quantum dots


Quantum dots
============

Adapted from Jana's thesis.


Quantum dots are artificially structured systems confining charges within
regions small enough to make their quantum mechanical energy levels
observable (Hanson et al. 2007,  Ihn book chapter 18).
While it is a physical system made up of atoms, each having many electrons,
the relevant particles of interest are the free electrons which can be
manipulated and probed with electric fields.
Quantum dots confining a small number of free electrons can be formed
in many different systems, but the main system of interest for charge, spin and
topological qubits are gate-defined quantum dots.
Here, lithographically fabricated gate electrodes in proximity of a nanowire
or on top of the surface of a heterostructure are used to create a potential
landscape with wells and barriers. Together with the system's geometry, this
potential landscape confines charges in all three dimensions.  As a result of
this confinement and opposed to the continuum of states that typically occur
in macroscopic systems, quantum dots feature a discrete energy spectrum. For
this reason, quantum dots are sometimes called artificial atoms
(Kouwenhoven et al. 2001).
In what follows, we focus on quantum dots formed in two-dimensional
electron gas (2DEG) in a GaAs/AlGaAs heterostructure, illustrated in figure
:numref:`fig_2deg`.


Dots in a two-dimensional heterostructure (2DEG)
------------------------------------------------

.. _fig_2deg:
.. figure:: ./quantum_dots-18.svg
   :alt: A 2DEG heterostructure
   :align: center
   :width: 45.0%

   GaAs/AlGaAs heterostructure hosting a double quantum dot. Mobile electrons
   from the thin silicon donor layer are attracted by the lower energy arising
   due to a smaller band gap in GaAs, all while being held close to their
   ionised donors.  As a result, electrons are confined along the `z` axis,
   creating a two-dimensional electron gas (2DEG). Wells and barriers in the
   potential profile created by voltages applied to electrostatic gates
   further localize electrons within the 2DEG.

Mobile electrons from a thin silicon donor layer within  AlGaAs layer are
attracted by GaAs - AIGaAs  interface, marked in red in figure
:numref:`fig_2deg`. This is due
to the lower energy (energy profile depicted in black) they carry there. This
effectively confines charges on a 2D plane.

Control of the charges is achieved by applying voltages to electrostatic gates
at the surface of the heterostructure. The electric field created by these
gates repels electrons from underneath, creating a potential profile with
barriers and wells and thus localising electrons. An illustration of
electron confinement in one or several directions is depicted in
:numref:`fig_dims`. By choosing appropriate gate voltages, electrons can be
confined in two directions, creating a one dimensional channel, or in all
three directions, resulting in an effectively zero dimensional system -
the quantum dot. Note that in the case of a nanowire, charges are already
confined in two directions and gate voltages only need to create barriers to
localize them in the third direction. If the one dimensional confinement is
very narrow, with a width comparable to the electron's wavelength, the system
is called a quantum point contact (QPC).

.. _fig_dims:
.. figure:: ./quantum_dots-22.svg
    :alt: Dimensions of electron gases.
    :align: center
    :width: 50.0%

    Dimensions of electron gases.


Transport through quantum dots
------------------------------

The physics of quantum dots can be studied based on the dots' transport
properties, i.e. by measuring current through the system.
To this end, a small bias voltage is applied to metallic reservoirs on both
sides of the quantum dot device, allowing electrons to move between
reservoirs and dots via tunnelling processes. Current probes attached to the
reservoirs are used to measure the resulting current.
The fundamental transport phenomenon in quantum dots is the Coulomb
blockade (Van Houten et al 1992). The Coulomb blockade is a classical effect
arising due to the Coulomb repulsion between electrons, resulting in a
finite energy cost when adding an extra electron onto a dot.  At low enough
temperatures, tunnelling of electrons between dots or dots and adjacent
reservoirs can be suppressed and the device's current-voltage relation no
longer follows Ohm's law.

Once formed, quantum dots can be modelled as conducting islands connected to
source (S) and drain (D), illustrated in :numref:`fig_single_dot_model` and
:numref:`fig_double_dot_model`. Here, the conducting island is coloured grey,
while source and drain are shown in blue. Couplings between
dots and reservoirs are modelled as a resistor in parallel to a capacitor.

.. _fig_single_dot_model:
.. figure:: ./quantum_dots-19.svg
   :alt: Simplified model of a single dot.
   :align: center
   :width: 45.0%

   Schematic of a single dot, modelled as a conducting island connected via
   tunnel junctions to source and drain reservoirs. A nearby plunger gate is
   capacitively coupled to the dot and used to tune the its energy levels.

.. _fig_double_dot_model:
.. figure:: ./quantum_dots-20.svg
   :alt: Simplified model of a double dot.
   :align: center
   :width: 45.0%

   Schematic of a double dot, modelled as two conducting islands in series with
   source and drain reservoirs, coupled by tunnel junctions.


When a quantum dot is in the Coulomb blockade regime, transport through the
dot-reservoir system only occurs when the dot's energy level :math:\epsilon
falls between the energy levels of the reservoirs held at different bias
potentials, also called the bias window.

Transport through a double dot system is illustrated in figures
:numref:`fig_2d_hop_allowed` and :numref:`fig_2d_hop_forbidden`.
A  measurable current arises when one or both dot energy levels are within the
energy levels of the reservoirs (fig. :numref:`fig_2d_hop_allowed`). To be
precise, resonant tunnelling occurs when both dot levels are within the bias
window. A so-called co-tunnelling process via virtual states takes place when
only one potential is within the bias window (De Franceschi et al. 2001). The
resonant current is typically higher in amplitude than the off-resonant current.
The double-dot states are probed by stepping over the voltages of two
nearby gates, resulting (in the most typical and ideal case) in the charge
transition pattern illustrated in :numref:`fig_chargediagram`.

Note that one often omits the small bias and talks about the alignment of the
dots energy levels.
If the levels don't align, as in :numref:`fig_2d_hop_forbidden`, the dot
is in the Coulomb blockade regime. Here, Coulomb repulsion between electrons prevents
multiple electrons to occupy the same energy level and transport is suppressed.


.. _fig_2d_hop_allowed:
.. figure:: ./quantum_dots-07.svg
    :alt: Electron transport of double dot: aligned energy levels.
    :align: center
    :width: 45.0%

    Electron transport via resonant tunnelling occurs from source to drain when
    both dot energy levels are within the bias window.


.. _fig_2d_hop_forbidden:
.. figure:: ./quantum_dots-06.svg
    :alt: Electron transport of double dot: energy levels not aligned.
    :align: center
    :width: 45.0%

    Electron tunnelling is suppressed whenever the dots' energy levels are not
    within the energy levels of the reservoirs. However, if one energy level is
    within the bias window, a measurable current arises due to a so-called
    co-tunnelling processes via virtual states.

.. _fig_chargediagram:
.. figure:: ./quantum_dots-12.svg
    :alt: Schema of a charge diagram
    :align: center
    :width: 40.0%

    Schema of a so-called charge diagram showing the charge
    transition pattern of a double dot.

.. _fig_chargediag_explained:
.. figure:: ./quantum_dots-04.svg
    :alt:
    :align: center
    :width: 40.0%

    Illustration of charge transitions measured by varying both plunger gate
    voltages and monitoring current from source to drain. Four types of charge
    transitions are observed, exchanging charges between dots, a dot and its
    adjacent reservoir, or allowing a current to flow from source to drain.

Transport features of dots with well-localized and weakly coupled charges can
be explained and qualitatively reproduced by the classical capacitance model,
which represents gates, dots and reservoirs as conductors connected through
resistors and capacitors. It also allows to capture the so-called gate
cross-talk, i.e. the effect of capacitive couplings of all gates to each dot.
The capacitance model is discussed in :ref:`capa_model`.
