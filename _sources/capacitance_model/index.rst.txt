
.. _capa_model:

Capacitance model
=================

This section is an extraction from Jana's `thesis`_.

Quantum dots with weak coupling between them and thus well localized charges
separated from the remaining of the electron gas
can be modelled by the capacitance model, also referred to as the
constant-interaction model (van der Wiel et al. 2002). The constant-interaction model
is a classical description based on two assumptions (Hanson et al. 2007). First,
the Coulomb interactions between electrons on dots and in reservoirs are
parametrized by constant capacitances. Second, the single-particle energy-level
spectrum is considered independent of electron interactions and the number of
electrons, meaning that quantum mechanical energy spacings are not taken into
account.

The system of electrostatic gates, dots and reservoirs is represented by a
system of conductors connected via resistors and capacitors. Albeit classical
and simple, the capacitance model explains and qualitatively reproduces
relevant transport features of gate-defined quantum dots.
The following paragraphs are based on van der Wiel et al. 2002 and
discussions with John K. Gamble.

The capacitance model is defined by a capacitance matrix :math:`\mathbf{C}`,
whose elements :math:`C_{ij}` are the capacitances between individual
elements :math:`i` and :math:`j`.
We distinguish between two different types of elements, the charge and the
voltage nodes, representing quantum dots and electrostatic gates respectively.
Each node :math:`i` is defined by its charge :math:`Q_{i}` and electrical
potential :math:`V_{i}`. For simplicity, we write charges and potentials on
all nodes of the system in vector notation. We denote charges on charge and
voltages nodes by :math:`\vec{Q_{c}}` and :math:`\vec{Q_{v}}` respectively,
and electrical potentials by :math:`\vec{V}_{c}` and :math:`\vec{V}_{v}`.
The capacitance model allows to calculate potentials on voltage nodes
resulting in the desired number of charges on charge nodes. We consider a
system of :math:`N_{c}` charge nodes and :math:`N_{v}` voltage nodes.

The capacitor connecting node :math:`j` and node :math:`k` has a
capacitance :math:`C_{jk}` and stores a charge :math:`q_{jk}`. The total
charge on node :math:`j` is the sum of the charges of  all capacitors
connected to it,

.. math::

   Q_{j} = \sum_{k} q_{jk} = \sum_{k} C_{jk} (V_{j} - V_{k}).

Using the vector notation for charges and electrical potentials introduced
above, this relation can be expressed using the capacitance matrix,

.. math::
   :label: fullcapa

   \vec{Q} = \mathbf{C} \vec{V}.

Distinguishing between charge and voltage node sub-systems, this relation
becomes

.. math::
   :label: capa_general

   \begin{pmatrix}
      \vec{Q_{c}} \\
      \vec{Q_{v}}
   \end{pmatrix} =
   \begin{pmatrix}
      \mathbf{C_{cc}} & \mathbf{C_{cv}} \\
      \mathbf{C_{vc}} & \mathbf{C_{vv}}
   \end{pmatrix}
   \begin{pmatrix}
      \vec{V_{c}} \\
      \vec{V}_{v}
   \end{pmatrix}.

Diagonal elements of the capacitance matrix, :math:`C_{jj}`, are total
capacitances of each node and carry the opposite sign of the matrix's
off-diagonal elements.
The off-diagonal elements of :math:`\mathbf{\mathbf{C_{cc}}}` are
capacitances between charge nodes, while the off-diagonal elements of
:math:`\mathbf{\mathbf{C_{vv}}}` are capacitances between voltage nodes.
The elements of :math:`\mathbf{\mathbf{C_{cv}}}` are capacitances between
voltage and charge nodes, and allow to calculate so-called virtual gate
coefficients - useful knobs in semiconductor qubit experiments.

The capacitance matrix allows to derive an expression of the charge nodes'
electrochemical potential :math:`\mu`, which is defined as the energy
difference of two charge configurations and will be formally defined shortly.
The total energy of a system of conductors defined by a capacitance matrix
:math:`\mathbf{C}` and voltages :math:`\vec{V}` is given by

.. math::
   :label: engen

   U = \frac{1}{2} \vec{V}^{T} \mathbf{C} \vec{V}.

Using :eq:`capa_general`, we can express the charge nodes' potentials
:math:`\vec{V}_{c}` as

.. math::
   :label: Vc

   \vec{V_{c}} = \mathbf{C}^{-1}_{cc} \left( \vec{Q_{c}} - \mathbf{C_{cv}} \vec{V}_{v} \right).


Combining :eq:`Vc` and :eq:`engen`, the energy reads

.. math::
   :label: energy

   U = \frac{1}{2} \vec{Q_{c}}^{T} \mathbf{C^{-1}_{cc}} \vec{Q_{c}}
      + \frac{1}{2} \vec{V^{T}_{v}} \mathbf{C_{vc}} \mathbf{C_{cc}^{-1} } \mathbf{C_{cv}} \vec{V}_{v}
      - \vec{Q^{T}_{c}} \mathbf{C^{-1}_{cc}} \mathbf{C_{cv}} \vec{V}_{v}.


We now assume that the number of charges on charge nodes :math:`\vec{Q_{c}}`
and potentials at voltages nodes :math:`\vec{V}_{v}` to be known.
In experiments, this assumption is valid in the few electron regime. We
substitute the charge node charge vector by the corresponding a vector
containing the number of electrons. Let

.. math::

   \vec{N} = \frac{- \vec{Q}_{c}}{q},

where :math:`q` is the elementary charge.
In general, the electrochemical potential is defined as the energy difference
between the states  :math:`\vec{N}` and  :math:`\vec{N} + \hat{e}_{j}`,
where  :math:`\hat{e}_{j}` is a unit vector indicating an additional electron
on charge node :math:`j`.
The electrochemical potential of a charge node :math:`j` thus becomes

.. math::
   :label: mu

   \mu_{j}(\vec{N}, \vec{V}_{v}) = U(\vec{N}, \vec{V}_{v}) - U( \vec{N} - \hat{e}_{j}, \vec{V}_{v}).



Using this definition together with :eq:`energy` we obtain

.. math::
   :label: potential

   \mu_{j}(\vec{N}, \vec{V}_{v}) &= \frac{q^{2}}{2} \vec{N}^{T}  \mathbf{C_{cc}^{-1}} \vec{N} + q \vec{N}^{T}  \mathbf{C_{cc}^{-1}}  \mathbf{C_{cv}} \vec{V}_{v}
				- \frac{q^{2}}{2} \left( \vec{N} - \hat{e}_{j} \right)^{T}  \mathbf{C_{cc}^{-1}} \left( \vec{N} - \hat{e}_{j} \right)
				- q \left( \vec{N} - \hat{e}_{j} \right)^{T} \mathbf{C_{cc}^{-1}}  \mathbf{C_{cv}} \vec{V}_{v} \\
			&= \frac{-q^{2}}{2} \hat{e}_{j}^{T}  \mathbf{C_{cc}^{-1}} \hat{e}_{j}
				+ q^{2} \vec{N}^{T}  \mathbf{C_{cc}^{-1}} \hat{e}_{j}
				+ q \hat{e}_{j}^{T}  \mathbf{C_{cc}^{-1}}  \mathbf{C_{cv}} \vec{V}_{v}.


For simplicity, we assume the electrochemical potentials of source and drain to
be zero. This assumption is reasonable for cases when the bias between source
and drain is infinitesimal.
In this case, a current arises when the electrochemical potentials of all dots
are zero and no energy is required to add another electron.
There are two possible conditions for electrochemical potentials to be zero,
for each dot and each charge configuration,

.. math::
   :label: tp_cond_general

   \mu_{j}{(\vec{N}, \vec{V}_{v})}  &=0 \quad \forall j,    \Rightarrow \text{electron transport}\\
   \mu_{j}{(\vec{N} + \hat{e}_{j}, \vec{V}_{v})}  &= 0 \quad \forall j.   \Rightarrow \text{hole transport},


Using the expression :math:`\mu` in :eq:`potential`, these conditions become

.. math::
   :label: tp_conditions

   \text{Electron:} \quad
   0 &= - \frac{q^{2}}{2} \hat{e}_{j}^{T}  \mathbf{C_{cc}^{-1}} \hat{e}_{j}
      + q^{2} \vec{N}^{T}  \mathbf{C_{cc}^{-1}} \hat{e}_{j}
      + q \hat{e}_{j}^{T}  \mathbf{C_{cc}^{-1}}  \mathbf{C_{cv}} \vec{V}_{v} \quad \forall j, \\
      \text{Hole:} \quad
      0 &= -\frac{q^{2}}{2} \hat{e}_{j}^{T}  \mathbf{C_{cc}^{-1}} \hat{e}_{j}
      + q^{2} \left( \vec{N} + \hat{e}_{j} \right)^{T}  \mathbf{C_{cc}^{-1}} \hat{e}_{j}
      + q \hat{e}_{j}^{T}  \mathbf{C_{cc}^{-1}}  \mathbf{C_{cv}} \vec{V}_{v} \quad \forall j.


These equations can be used to either calculate potentials, i.e. gate voltage
combinations, resulting in a particular charge configuration and hence
determine voltages at which charge transitions occur, or to extract the
capacitance matrix if enough charge transitions are known.
However, a system of :math:`N` nodes consists of  :math:`N(N-1)/2`
capacitors and thus requires :math:`N(N-1)/2` many relations to fully
determine :math:`\mathbf{C}`.

Double quantum dots
-------------------

We now consider the specific case of  two charge and six voltage nodes,
representing common 2DEG device layouts  for semiconductor
qubits.
An illustration of the layout as well as the corresponding capacitance model
are shown in :numref:`fig_device_scheme` and :numref:`fig_capa_coupling_device2`
respectively. We denote the two
charge nodes, i.e. dots, by capital letters :math:`A` and :math:`B` and
voltages nodes, i.e. gates, by numerical Indices between 0 and 5.

.. _fig_device_scheme:
.. figure:: ../quantum_dots/quantum_dots-09.svg
    :alt: Schema double dot device.
    :align: center
    :width: 30.0%

    Schema of a double dot device with six electrostatic gates.

.. _fig_capa_coupling_device2:
.. figure:: ../quantum_dots/quantum_dots-08.svg
    :alt: Capacitive coupling between gate electrodes and dots of a 2D double dot device.
    :align: center
    :width: 40.0%

    Capacitive coupling between gates and gates and dots of a double dot device
    with six electrostatic gates.


The capacitance sub-matrices of this system are

.. math::

   \mathbf{C}_{cc} =
   \begin{bmatrix}
      C_{A} & C_{m} \\
      C_{m} & C_{B}
   \end{bmatrix},
   \quad
   \mathbf{C}_{cv}  =
   \begin{bmatrix}
      C_{A0} & C_{A1} & C_{A2} & C_{A3} & C_{A4} & C_{A5} \\
      C_{B0} & C_{B1} & C_{B2} & C_{B3} & C_{B4} & C_{B5}
      \end{bmatrix},

where :math:`C_{m}` is the inter-dot capacitance.  :math:`C_{A}` and
:math:`C_{B}` are the sum of all capacitances connected to :math:`A` and :math:`B`,


.. math::

   C_{A} &= \sum_{k=0,..,5} C_{Ak} + C_{m} + C_{S} \nonumber \\
   C_{B} &= \sum_{k=0,..,5} C_{Bk} + C_{m} + C_{D}.

Here :math:`C_{S}` and :math:`C_{D}` are capacitances between :math:`A` and
source, and :math:`B` and drain respectively.  These capacitances can be
determined from measurements probing the charge states of the system.

Charge diagram
--------------

The charge diagram, sometimes also called the charge stability diagram, is a
two-dimensional measurement stepping over two gate voltages while probing the
dots' stable electron configurations. As introduced above, charge transitions
of a double quantum dot form hexagonal domains. The shape and dimensions of
these domains depend, among others, on the capacitive coupling between gates
and dots.

.. _fig_charge_diagrams_a:
.. figure:: ../quantum_dots/quantum_dots-12.svg
   :alt:
   :align: center
   :width: 35.0%

   Characteristic honeycomb pattern of moderately coupled quantum dots. The
   two types of triple points are marked by red and blue dots.

.. _fig_charge_diagrams_b:
.. figure:: ../quantum_dots/quantum_dots-15.svg
   :alt:
   :align: center
   :width: 35.0%

   Electron and hole triple points. The terminology originates from the fact
   that transport through the reservoir-dots system can be viewed as either
   electron or hole tunnelling events. At triple points marked in blue, an
   electron is tunnelling counter-clockwise, while at the triple points
   marked in red a hole is tunnelling clock-wise.

.. _fig_charge_diagrams_c:
.. figure:: ../quantum_dots/quantum_dots-10.svg
   :alt:
   :align: center
   :width: 35.0%

   Charge diagram of a double quantum dot with a vanishing inter-dot
   capacitance and capacitive coupling to distant plungers.

.. _fig_charge_diagrams_d:
.. figure:: ../quantum_dots/quantum_dots-11.svg
    :alt:
    :align: center
    :width: 35.0%

    Charge diagram of a double quantum dot with a vanishing inter-dot
    capacitance but non-zero coupling to distant plungers, also referred to
    as cross-talk between gates.


For double quantum dots, the two general conditions of transport to occurs is
given by :eq:`tp_cond_general` and result in charge degeneracy points called
triple points. We here assume that the voltage difference between source and
drain is infinitesimal and thus negligible in our derivation. Triple points
come in two flavours, which are often viewed as hole and electron transfer
processes and illustrated in  :numref:`fig_charge_diagrams_b`.
The lower left triple point can be viewed as electrons tunnelling
counter-clockwise, while the upper right as a hole tunnelling clockwise.
In this picture, the double dot system cycles through the following charge
states voltage combinations:

.. math::

   \text{Electrons:} \quad  & (N_{A}, N_{B}) \rightarrow (N_{A}+1, N_{B}) \rightarrow (N_{A}, N_{B}+1) \rightarrow (N_{A}, N_{B})  \nonumber

   \text{Holes:} \quad  & (N_{A}+1, N_{B}+1) \rightarrow (N_{A}+1, N_{B}) \rightarrow (N_{A}, N_{B}+1) \rightarrow (N_{A}+1, N_{B}+1)


The dimensions and shape of the honeycomb cells depend on the strength of the
capacitive coupling between dots and gates.  :numref:`fig_charge_diagrams_c`
shows a diagram of a system where each plunger tunes a single dot only.
This is sometimes referred to as a system with no cross-capacitances,
meaning that the capacitive coupling between dots and distant gates is
negligible.  :numref:`fig_charge_diagrams_d` shows a more
realistic scenario, where both plunger gates are coupled to each dot,
resulting in inclined charge transitions. In both of these examples, the
inter-dot capacitance is negligible, which results in a vanishing spacing
between triple points.

.. _fig_honey_spacing:
.. figure:: ../quantum_dots/quantum_dots-16.svg
    :alt:
    :align: center
    :width: 35.0%

    Honeycomb pattern with relevant voltage spacings. The geometry of a
    honeycomb cell is directly related to the capacity coupling between dots
    and gates.


Let us assume the reservoirs' electrochemical potentials to be zero and that
one varies a single voltage :math:`\Delta_{k}` of a gate :math:`k`  to measure
two triple points of the same kind (electron or hole). Both triple points occur
when the electrochemical potentials of the respective charge configurations
vanish, and thus

.. math::
   :label: electron_dV

   \mu_{j}(\vec{N}, \vec{V}_{v}) = \mu_{j}(\vec{N}+\hat{e}_j, \vec{V}_{v} + \Delta_{jk} \hat{e}_{k}), \quad \forall j, \forall k.


Using the expression of the electrochemical potential in :eq:`potential`, we
can relate distances in voltage space to capacitive couplings between dots
and gates. Specifically, we obtain that

.. math::
   :label: first_rel

   \Delta_{jk} = \frac{-q}{C_{jk}}.


As a concrete example, the double-dot system's two plunger gate voltages
:math:`V_{2}` and :math:`V_{4}` are varied while  all other voltages fixed are
kept fixed.
For clarity, we omit fixed voltages and express the dots' charge vector
explicitly. In this notation, :eq:`electron_dV` for the double dot system reads

.. math::

   \mu_{A}(N_{A}, N_{B}; V_{2}, V_{4})  & = \mu_{A}(N_{A}+1, N_{B}; V_{2}+\Delta_{2}, V_{4}) \nonumber

   \mu_{B}(N_{A}, N_{B}; V_{2}, V_{4})  & = \mu_{B}(N_{A}, N_{B}+1; V_{2}, V_{4}+\Delta_{4}).

The voltage spacings are related to the dot-gate capacitances by

.. math::

   \Delta_{A2}  = \frac{-q}{C_{A2}}, \quad \Delta_{B4} = \frac{-q}{C_{B4}}

and  illustrated in  :numref:`fig_honey_spacing`. These relations allow to
determine two entries of the capacitance matrix :math:`\mathbf{C_{cv}}`.
The condition relating electron to hole triple points reads

.. math::

   \mu_{j}(\vec{N}, \vec{V}_{v}) = \mu_{j}(\vec{N} + \hat{e}_{l}, \vec{V}_{v} + \Delta^{m}_{jlk} \hat{e}_{k}), \quad \forall j,l, k, \quad l \neq j.

Again, using :eq:`potential`, we are able to relate capacitances to voltage
spacings as follows,

.. math::
   :label: second_rel

   \Delta_{jlk}^{m} = \frac{-q (\mathbf{C^{-1}_{cc}})_{lj}}{(\mathbf{C_{cc}^{-1}}\mathbf{C_{cv}})_{jk}}.

In our specific double-dot case with all fixed voltages omitted, we obtain

.. math::

   \mu_{A}(N_{A}, N_{B}; V_{2}, V_{4})  = \mu_{A}(N_{A}, N_{B}+1; V_{2}+\Delta_{AB2}^{m}, V_{4}) \nonumber \\

   \mu_{B}(N_{A}, N_{B}; V_{2}, V_{4})  = \mu_{B}(N_{A}+1, N_{B}; V_{2}, V_{4}+\Delta_{BA4}^{m}),

such that

.. math::

   \Delta_{AB2}^{m} = \frac{q C_{m}}{C_{B}C_{A2} - C_{m}C_{B2}}

   \Delta_{BA4}^{m} = \frac{q C_{m}}{C_{A}C_{B4} - C_{m}C_{A4}}

Note that these equations are the same as in Ref.[vanderWiel]_, but with
:math:`C_{B2} \neq 0` and  :math:`C_{A4} \neq 0`.
With an appropriate series of two-dimensional measurements sweeping over
distinct gate combinations, these equations allow to extract the entries of
both :math:`\mathbf{C_{cv}}` and :math:`\mathbf{C_{cc}}`.

Note on synthetic data
----------------------

The model described above was used to generate one part of the synthetic dataset
used in `Darulova et al. 2021`_.
The general problem with synthetic data or simple models such as the one above is
that their ability to reproduce real device behavior is limited. Here fro example,
only two out of many possible double dot states are covered. The situation is better for single dots as they are a lot
simpler and diagonal lines is (almost) all there is for good versions of this regime.
The biggest shortcoming of the model is that it only generates
well-defined dots, without a suitable way to reproduce 'poor' regimes or regimes
where the dots are not well defined. They don't allow to test tuning sequences aiming to
tune away from poor or no-dot regimes.
What the current implementation does allow however, is to sweep arbitrary
gates of an N-dot system. It thus reproduces gate cross-talk, which manifests
itself in the shift of transport features if gate voltages of a nearby gate is changed.


.. _thesis: https://doi.org/10.3929/ethz-b-000473107

.. _Darulova et al. 2021: https://iopscience.iop.org/article/10.1088/2632-2153/ac104c
