Nanotune is a QCoDeS- and Python-based automated tuning software, which uses 
supervised machine learning to enable autonomous bring-up of gate-defined 
quantum dots.

What it does
============

Defining quantum dots in semiconductor-based heterostructures is an essential 
step in initializing solid-state qubits. With growing device complexity and 
increasing number of functional devices required for measurements, a manual 
approach to finding suitable gate voltages to confine electrons electrostatically 
is impractical. 

Nanotune automates typical manual measurements and replaces 
the exerimenter's decision about next tuning steps by supervised machine learning.
It has enabled the implementation of a two-stage device characterization and 
dot-tuning process, which first determines whether devices are functional and 
then attempts to tune the functional devices to the single or double quantum-dot 
regime. Measurement quality assessement and charge state detection on charge stability 
diagrams is done using four binary classifiers trained with experimental data, reflecting 
real device behavior.

While autonomous tuning has been demonstrated on spin qubit devices in GaAs, published
`here <https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.13.054005>`_, 
nanotune can easly be used to implement tuning sequences for other devices structures and materials.

Nanotune also provides a data processing pipeline starting from data aquisition to machine learning algorihtm application.
This allowed to conduct the comparison between synthetic and experimental training data detailed
`here <https://arxiv.org/abs/2005.08131>`_.

Word of caution
===============
The code in this repo is still under developement. The following sub-modules are not stable:

- labelling/*
- model/*
- classification/*
- device/deviceconfigurator
- device_tuner/tuningreport
- device_tuner/fivedottuner

Submodules which are likely to change in the future are:

- device/device
- device_tuner/tuner
- data/dataset


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

Code of Conduct
---------------

nanotune strictly adheres to the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`__. 

Acknowledgments
---------------

A special thanks goes to Matthias Troyer, Maja Cassidy, David Reilly and Charles Marcus for initiating, supervising and pushing over the finish line the PhD project, which resulted in the original version of nanotune. Chris Granade, Nathan Wiebe, John Hornibrook - your inputs were invaluable in implementing several modules. Alice Mahoney, Sebastian Pauka, Rachpon Kalra - thank you for setting up fridges and prepping devices, making it possible for automated tuning to be developed, tested and demonstrated. William H.P. Nielsen and Jens Nielsen - both your patience and help with qcodes was essential.
