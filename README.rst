**Nanotune is not maintained and has been archived**


Nanotune is a QCoDeS- and Python-based automated tuning software, which uses
supervised machine learning to enable autonomous bring-up of gate-defined
quantum dots.

Read more about it `here <https://microsoft.github.io/nanotune/overview/index.html>`_.

Getting Started
===============

Requirements
------------
nanotune requires a working python 3.7+ installation. We recommend using Anaconda or Miniconda for managing nanotune's python environment.

Installing
----------

At the moment, nanotune is only available from GitHub. After cloning, you can set up a python environment in your preferred way, either using the requirements.txt or environment.yml. When using conda, simply navigate into the outer nanotune folder in a terminal and type:

.. code::

    conda env create -f environment.yml
    conda activate nanotune
    pip install -e .


License
-------

See `License <https://github.com/microsoft/nanotune/blob/main/LICENSE>`__.

Code of Conduct
---------------

nanotune strictly adheres to the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`__.
