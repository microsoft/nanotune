# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest
from nanotune.device.device import NormalizationConstants


def test_normalization_constant_init():
    norm = NormalizationConstants()
    assert sorted(norm.__dataclass_fields__.keys()) == [
        'rf', 'sensing', 'transport',
    ]
    assert norm.transport == (0., 1.)
    assert norm.sensing == (0., 1.)
    assert norm.rf == (0., 1.)


def test_normalization_constants_update():
    norm = NormalizationConstants()
    norm.update({"transport": [-2, -1]})
    assert norm.transport == (-2, -1)
    assert norm.sensing == (0., 1.)
    assert norm.rf == (0., 1.)

    norm.update(NormalizationConstants(transport=[-1.1, 0.5], sensing=(2, 1)))
    assert norm.transport == (-1.1, 0.5)
    assert norm.sensing == (2, 1)
    assert norm.rf == (0., 1.)

    with pytest.raises(ValueError):
        norm.update([-1.1, 0.5])

    with pytest.raises(TypeError):
        norm.update({'transport': -1})

    with pytest.raises(KeyError):
        norm.update({'dc_transport': (0, 1)})
