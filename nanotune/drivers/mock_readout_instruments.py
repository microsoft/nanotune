# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import qcodes as qc


class MockLockin(qc.Instrument):

    def __init__(
            self,
            name: str,
            **kwargs):
        super().__init__(name=name, **kwargs)

        self.add_parameter("frequency",
                           parameter_class=qc.Parameter,
                           initial_value=125.,
                           unit='Hz',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("amplitude",
                           parameter_class=qc.Parameter,
                           initial_value=0.,
                           unit='V',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("phase",
                           parameter_class=qc.Parameter,
                           initial_value=0.,
                           unit='deg',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("time_constant",
                           parameter_class=qc.Parameter,
                           initial_value=1.e-3,
                           unit='s',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("Y",
                           parameter_class=qc.Parameter,
                           initial_value=0.,
                           unit='V',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("X",
                           parameter_class=qc.Parameter,
                           initial_value=0.,
                           unit='V',
                           get_cmd=None, set_cmd=None)


class MockRF(qc.Instrument):

    def __init__(
            self,
            name: str,
            **kwargs):
        super().__init__(name=name, **kwargs)
        self.add_parameter("frequency",
                           parameter_class=qc.Parameter,
                           initial_value=530e6,
                           unit='Hz',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("power",
                           parameter_class=qc.Parameter,
                           initial_value=7.,
                           unit='dBm',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("phase",
                           parameter_class=qc.Parameter,
                           initial_value=0.,
                           unit='deg',
                           get_cmd=None, set_cmd=None)
        self.add_parameter("status",
                           parameter_class=qc.Parameter,
                           initial_value=False,
                           get_cmd=None, set_cmd=None)