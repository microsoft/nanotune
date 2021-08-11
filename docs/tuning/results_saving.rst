.. _results_saving:


Results saving
==============

Three classes work together to ensure tuning results are retained:
`TuningResult`, `MeasurementHistory`, `MeasurementHistory`. They can be used to
also save any other, not tuning related, results if desired.


TuningResult
------------

The `TuningResult` is a dataclass holding results of a tuning stage such as
gate characterization or charge diagram. It can also be used without
nanotune's `Tuningstage` and is serializable.
Its attributes keep track of important metadata such as the success of the
stage or measurement,
guid/data run IDs and where the data is saved. The latter is important when
several databases are used.


MeasurementHistory
------------------

The `MeasurementHistory` is a container holding tuning results of a single
device. Its `results` attribute is a dictionary mapping string identifiers
such as `gate_characterization_top_barrier` to a `TuningResult` instances. If
appearing multiple times, these
identifiers are made unique by appending numbers (as strings) at the end.


TuningHistory
-------------

The `TuningHistory` class is a container for
tuning results of multiple devices. It wraps a dictionary mapping a device
name to an instance of a `MeasurementHistory`.
