.. _config:

Configuration
=============

The configuration system is a very light version of the one used by QCoDeS.
It simply loads the content of `config.json` to a dictionary whose content
can be set or gotten via `nt.config`.  This file also defines most constant
such how key of nanotune metadata or machine labels need to be called.

To get the current database name and how the metadata key is called type:

.. code-block:: python

    import nanotune as nt

    nt.config['db_name']
    nt.config['core']['meta_add_on']

Note that setting `nt.config['db_name']` directly is not passed to qcodes and
`nt.set_database` should be used
to set a default database instead.
Nothing prevents from the content of `nt.config` to be changes right now.
QCoDeS needs to be configured separately.
