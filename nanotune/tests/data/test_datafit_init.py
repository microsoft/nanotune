import pytest

import nanotune as nt
from nanotune.fit.datafit import DataFit


def test_datafit_init():
    with pytest.raises(TypeError):
        DataFit(1, "temp.db")
