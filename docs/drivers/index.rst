
.. _drivers:

Instruments & interfaces
========================


Interfaces
----------

The idea of using interfaces between the hardware used and
nanotune is to keep the automation code general and independent of the
instruments. It is up to the user to specify the methods getting and setting
parameters by sub-classing the provided interfaces. There are two of them:
the `DACInterface`, for a DAC instrument, and `DACChannelInterface`, for the
DAC channel. It is the `DACInterface` subclass that needs to be used in
measurements and its channels passed to the init method of `DeviceChannel`.

Specifically, `DACInterface` establishes that its channel list is called
`channels` and each channel is referred to by `chXY`, where XY is a
two-digit number with a leading zero for single digits.
The `DACChannelInterface` defines the voltage to be set and gotten by `set_voltage`
and `get_voltage` respectively, and for the channel number to be specified by
the `channel_id` property. The user also has to supply information about whether
the channel's voltage can be ramped by using a software ramp, by implementing the
`supports_hardware_ramp` property. Other setter and getter methods similar
define the remaining functionalities.

An example on how to use these interfaces are `MockDACChannel` and `MockDAC`,
whose API documentation can be found in :ref:`drivers_api`.


Mock instruments
----------------

`MockDACChannel`, `MockDAC`, as well as `MockLockin` and `MockRF` are primarily
used in unit and functional tests, but can also appear in offline demos.


Server-client setup for DACs
----------------------------

In QCoDeS, it is currently not possible to connect to the same instrument from
the same python kernel, i.e. jupyter notebook. Sometimes it is convenient to
do exactly that, such as when measuring multiple devices in parallel.
A server-client setup is one option to enable such naive parallelism. In such
a setup, a server connected to the physical instrument in one python kernel
communicates with multiple other kernels via (string) messages. In the case of
a DAC, these messages say which parameter of a channel should be changed.
The `DACChannelServer` and `DACClient` are examples of how this server-client
message passing can be implemented. It should be taken more as an inspiration,
knowing that it is currently untested. No guarantee that it will not
blow up a device.
