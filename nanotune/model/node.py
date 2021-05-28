import logging
from typing import Optional

import qcodes as qc
from qcodes import validators as vals
from qcodes.utils.validators import Validator
from qcodes import Instrument, InstrumentChannel, Parameter
from qcodes.instrument.base import InstrumentBase

import nanotune as nt

logger = logging.getLogger(__name__)


class Node(InstrumentChannel):
    def __init__(
        self,
        parent: InstrumentBase,
        name: str = "node",
        label: str = "",
        node_type: Optional[str] = None,
        n_init: int = 0,
        v_init: float = 0,
    ):

        pass

        super().__init__(parent, name)

        self.add_parameter(
            "node_type",
            label="node type " + label,
            unit=None,
            get_cmd=None,
            set_cmd=None,
            initial_value=node_type,
            vals=vals.Strings(),
        )
        self.add_parameter(
            "n",
            label="number of charges " + label,
            unit=None,
            set_cmd=self._set_n,
            get_cmd=self._get_n,
            initial_value=n_init,
            vals=vals.Ints(-100000, 100000),
        )
        self.add_parameter(
            "v",
            label="voltage node " + label,
            unit="V",
            set_cmd=self._set_v,
            get_cmd=self._get_v,
            initial_value=v_init,
            vals=vals.Numbers(-100000, 100000),
        )

    def _set_n(self, new_N: int) -> None:
        self._n = new_N

    def _get_n(self) -> int:
        return self._n

    def _set_v(self, new_V: int) -> None:
        self._v = new_V

    def _get_v(self) -> float:
        return self._v
