import json
import logging
from abc import ABC, abstractmethod
from sqlite3 import OperationalError
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import xarray as xr
import qcodes as qc
from qcodes.dataset.experiment_container import load_by_id

import nanotune as nt
from nanotune.data.dataset import (Dataset, default_coord_names,
                                   default_readout_methods)

logger = logging.getLogger(__name__)
AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]


class DataFit(ABC, Dataset):
    def __init__(
        self,
        qc_run_id: int,
        db_name: str,
        db_folder: Optional[str] = None,
    ) -> None:

        if db_folder is None:
            db_folder = nt.config["db_folder"]

        Dataset.__init__(
            self,
            qc_run_id,
            db_name,
            db_folder=db_folder,
        )
        self._features: Dict[str, Any] = {}
        self._range_update_directives: List[str] = []

    @property
    def features(self) -> Dict[str, Any]:
        """"""
        if not self._features:
            self.find_fit()
        return self._features

    @abstractmethod
    def find_fit(self) -> None:
        """"""

    @property
    def range_update_directives(self) -> List[str]:
        """"""
        return self._range_update_directives

    def save_features(self) -> None:
        """"""
        nt.set_database(self.db_name, db_folder=self.db_folder)
        ds = qc.load_by_run_spec(captured_run_id=self.qc_run_id)
        try:
            nt_meta = json.loads(ds.get_metadata(nt.meta_tag))
        except (RuntimeError, TypeError, OperationalError):
            nt_meta = {}
        nt_meta["features"] = self.features
        ds.add_metadata(nt.meta_tag, json.dumps(nt_meta))

    def get_edge(
        self,
        which_one: str,
        readout_method: str = "transport",
        delta_v: float = 0.05,
        use_raw_data: bool = False,
    ) -> np.array:
        """"""
        if self.dimensions[readout_method] != 2:
            logger.error("get_edge method not implemented for 1D data.")
            raise NotImplementedError
        if use_raw_data:
            data = self.raw_data[self.readout_methods[readout_method]]
            coord_names = [str(it) for it in list(self.raw_data.coords)]
        else:
            data = self.data[readout_method]
            coord_names = default_coord_names["voltage"]

        signal = data.values
        voltage_x = data[coord_names[0]].values
        voltage_y = data[coord_names[1]].values

        if which_one == "left vertical":
            v_start = voltage_x[0]
            dv = 0
            didx = 1
            for ii in range(1, len(voltage_x)):
                dv += abs(v_start - voltage_x[ii])
                v_start = voltage_x[ii]
                if dv <= delta_v:
                    didx = ii
                else:
                    break
            edge = np.average(signal[:, 0 : didx + 1], axis=1)

        elif which_one == "bottom horizontal":
            v_start = voltage_y[0]
            dv = 0
            didx = 1
            for ii in range(1, len(voltage_y)):
                dv += abs(v_start - voltage_y[ii])
                v_start = voltage_y[ii]
                if dv <= delta_v:
                    didx = ii
                else:
                    break
            edge = np.average(signal[0 : didx + 1, :], axis=0)

        elif which_one == "right vertical":
            v_start = voltage_x[-1]
            dv = 0
            didx = 1
            for ii in range(1, len(voltage_x)):
                dv += abs(v_start - voltage_x[-ii])
                v_start = voltage_x[-ii]
                if dv <= delta_v:
                    didx = ii
                else:
                    break
            edge = np.average(signal[:, -(didx + 1) :], axis=1)

        elif which_one == "top horizontal":
            v_start = voltage_y[-1]
            dv = 0
            didx = 1
            for ii in range(1, len(voltage_y)):
                dv += abs(v_start - voltage_y[-ii])
                v_start = voltage_y[-ii]
                if dv <= delta_v:
                    didx = ii
                else:
                    break
            edge = np.average(signal[-(didx + 1) :, :], axis=0)

        else:
            logger.error("Trying to get an edge which does not exist:" + which_one)

        return edge

    @abstractmethod
    def plot_fit(self) -> AxesTuple:
        pass
