# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import qcodes as qc
from qcodes.instrument.delegate.delegate_instrument import DelegateInstrument
from qcodes.instrument.delegate.grouped_parameter import GroupedParameter

from nanotune.device.device import ReadoutMethods


def test_readout_methods_init(station):
    rm = ReadoutMethods()
    assert sorted(rm.__dataclass_fields__.keys()) == [
        'rf', 'sensing', 'transport',
    ]
    for field in rm.__dataclass_fields__.keys():
        assert getattr(rm, field) is None

    instr = DelegateInstrument(
        'dummy',
        station,
        parameters={'transport': 'lockin.phase'}
    )
    rm = ReadoutMethods(transport=instr.transport)
    assert isinstance(rm.transport, GroupedParameter)
    station.lockin.phase(34)
    assert rm.transport() == station.lockin.phase()

    rm = ReadoutMethods(transport=station.lockin.phase)
    assert isinstance(rm.transport, qc.Parameter)
    assert rm.transport() == station.lockin.phase()


def test_readout_methods_get_parameters(station, delegate_instrument):
    rm = ReadoutMethods(
        transport=delegate_instrument.test_param,
        sensing=station.rf.phase
    )
    params = rm.get_parameters()
    assert params == [delegate_instrument.test_param, station.rf.phase]


def test_readout_methods_as_name_dict(station, delegate_instrument):
    rm = ReadoutMethods(
        transport=delegate_instrument.test_param,
        sensing=station.rf.phase
    )
    params = rm.as_name_dict()
    assert params == {
        'transport': delegate_instrument.test_param.full_name,
        'sensing': station.rf.phase.full_name
    }


def test_readout_methods_available_readout(station, delegate_instrument):
    rm = ReadoutMethods(
        transport=delegate_instrument.test_param,
        sensing=station.rf.phase
    )
    params = rm.available_readout()
    assert params == {
        'transport': delegate_instrument.test_param,
        'sensing': station.rf.phase
    }
