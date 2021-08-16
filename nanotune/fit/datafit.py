import json
import logging
from abc import ABC, abstractmethod
from sqlite3 import OperationalError
from typing import Any, Dict, List, Optional, Tuple
import numpy.typing as npt
import matplotlib
import numpy as np
import qcodes as qc

import nanotune as nt
from nanotune.data.dataset import default_coord_names

logger = logging.getLogger(__name__)
AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]


class DataFit(ABC, nt.Dataset):
    """Abstract base class for fitting classes.
    It inherits from nt.Dataset to have access to post processed data as
    necessary. The abstract `find_fit` and `plot_fit` methods, as well as
    the `range_update_directives` property need to be
    implemented by child classes. They are used by tuningstage classes.

    Attributes:
        features: features extracted by child classes.
        range_update_directives: directives on how to update voltages other
            than those swept in the current fit. Typically indicating general
            directions such as more positive or negative, determined based
            on the normalized signal strength. E.g. if the signal is too low, i.e.
            the device is pinched off, this property will say that some other
            voltages need to be set to more positive values.

    """
    def __init__(
        self,
        qc_run_id: int,
        db_name: str,
        db_folder: Optional[str] = None,
        **kwargs,
    ) -> None:

        if db_folder is None:
            db_folder = nt.config["db_folder"]

        nt.Dataset.__init__(
            self,
            qc_run_id,
            db_name,
            db_folder=db_folder,
            **kwargs,
        )
        self._features: Dict[str, Any] = {}
        self._range_update_directives: List[str] = []

    @property
    def features(self) -> Dict[str, Any]:
        """Overwrites feature attribute of nt.Dataset class. Computes
        features if they are not found.

        Returns:
            dict: mapping strings indicating the feature type to its value.
        """
        if not self._features:
            self.find_fit()
        return self._features

    @property
    @abstractmethod
    def range_update_directives(self) -> List[str]:
        """Indicating general directions such as more positive or negative,
        determined based on the normalized signal strength. E.g. if the signal
        is too low, i.e. the device is pinched off, this property will say that
        some other voltages need to be set to more positive values.

        Returns:
            list: list of directives.
        """

    @abstractmethod
    def find_fit(self) -> None:
        """Method to extract features. To be defined by child classes and
        tailored to a specific measurement. It needs to populate self._features.
        """

    @abstractmethod
    def plot_fit(self) -> AxesTuple:
        """Should display a plot fit important features shown."""
        pass


    def save_features(self) -> None:
        """Saves extracted features to QCoDeS metadata."""
        nt.set_database(self.db_name, db_folder=self.db_folder)
        ds = qc.load_by_run_spec(captured_run_id=self.qc_run_id)
        try:
            nt_meta = json.loads(ds.get_metadata(nt.meta_tag))
        except (RuntimeError, TypeError, OperationalError):
            nt_meta = {}
        nt_meta["features"] = self.features
        ds.add_metadata(nt.meta_tag, json.dumps(nt_meta))
