.. _datafitting:

Data fitting
============

Base class
----------

The abstract `DataFit` class is the base for all fitting classes. It's main purpose
is to establish the presence of `find_fit` and `plot_fit` methods, as well as the
`next_actions` attribute required by :ref:`tuningstages`.

The `find_fit` method extracts features and determines the transport regime,
i.e. open, closed or intermediate. In case of a closed or open regime,
the `next_actions` list is populated with suggesting how voltages need to
be adjusted, e.g. more positive or negative.
The `plot_fit` method displays a plot with important features shown.

It also provides a method to save extracted features to metadata. Specifically,
the following code is used for this:

.. code-block:: python

    nt.set_database(self.db_name, db_folder=self.db_folder)
    ds = qc.load_by_run_spec(captured_run_id=self.qc_run_id)
    try:
        nt_meta = json.loads(ds.get_metadata(nt.meta_tag))
    except (RuntimeError, TypeError, OperationalError):
        nt_meta = {}
    nt_meta["features"] = self.features
    ds.add_metadata(nt.meta_tag, json.dumps(nt_meta))

More details about the features extracted for each measurement can be found in
:ref:`pinchoff`, :ref:`dotfit` and :ref:`coulomboscillationfit`.

.. toctree::
   :maxdepth: 2

   pinchofffit
   dotfit
   coulomboscillationfit
