=======
Drivers
=======

Interfaces
----------




Mock instruments
----------------

`MockDACChannel`, `MockDAC`, as well as `MockLockin` and `MockRF` are primarily
used in unit and functional tests. However, `MockDACChannel` and `MockDAC`
show how to use `DACChannelInterface` and `MockDAC`.


Server-Client Setup for DACs
----------------------------

In QCoDeS it is currently not possible to connect to the same instrument from
the same python kernel, i.e. jupyter notebook. Sometimes it is convenient to
do exactly that, such as when measuring multiple devices in parallel.
A server-client setup is one option to enable such naive parallelism. In such
a setup, a server, connected to the physical instrument in one python kernel,
communicates with multiple other kernels via (string) messages. In the case of
a DAC, these messages say which parameter of a channel should be changed.
The `DACChannelServer` and `DACClient` are examples of how this server-client
message passing can be implemented. It should be taken more as an inspiration,
knowing that it is currently untested. No guarantee that it will not
blow up a device.