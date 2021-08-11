.. _pinchoff:

Pinchoff
--------

The `PinchoffFit` fits a one dimensional I-V curve to a hyperbolic tangent to
determine the parameters of the tangent describing the measurements the best.
The fit function used is

.. math::

    a * (1 + d * (\tanh(b * v + c))),

where :math:`a` is the amplitude, :math:`b` the slope and :math:`c` the shift.
:math:`d` is the sign of tanh, which may be different for different types of
readout methods, e.g. rf. Note that there is a more sophisticated fit function
in the `sim` package.
These parameters as well as the residuals of the fit are retained as features.
Based on the first derivative of either the fit or normalized data, the active
range as well as transition voltage of the gate swept is determined. The
active/valid voltage range is generally indicated by :math:`[L, H]`, while
the transition voltage is indicated by a :math:`T`. These values, together with
the signal strength at each of these voltages is saved to metadata of the
(QCoDeS) dataset under the `nt.meta_tag` key.

Figures :numref:`pinchoff_fit` and :numref:`pinchoff_features` show the fit
together with extracted features. They have been plotted using the `plot_fit`
and `plot_features` methods.

.. _pinchoff_fit:
.. figure:: ./figs/pinchofffit_aaaaaaaa-0000-0000-0000-016c59f8305c.svg
    :alt: Pinchoff fit.
    :align: center
    :width: 60.0%

    Example of a pinchoff fit.

.. _pinchoff_features:
.. figure:: ./figs/pinchoff_features_aaaaaaaa-0000-0000-0000-016c59f8305c.svg
    :alt: Pinchoff features explained.
    :align: center
    :width: 60.0%

    Some of the pinchoff features explained.


Pinchoff labels
    To allow for supervised machine learning to determine the quality, pinchoff
    curves need to be labelled. nanotune uses to labels, good (1, True) and
    poor (0, False). In general, a good curve is one showing a clear transition
    between open and closed regime and a poor doesn't. However, there are many cases
    in-between, such as curves that don't start at max/open regime but slightly below
    or gates that pinch off in stages, with some noise, or with a small slope.
    It is up to the labeller to
    decide which types of imperfection belong to which category. Ideally,
    this decision
    is made beforehand, to ensure consistent labelling.
