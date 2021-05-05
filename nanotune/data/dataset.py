import os
import csv
import logging
import copy
import json
import xarray as xr
from math import floor
import sqlite3

import numpy as np

from scipy.ndimage import sobel, generic_gradient_magnitude
from typing import Optional, Dict, List, Tuple, Union, Mapping, Any

import scipy.signal as sg
import scipy.fftpack as fp
from scipy.ndimage import gaussian_filter

from sklearn.impute import SimpleImputer
from skimage.transform import resize

import qcodes as qc
from qcodes.dataset.data_set import new_data_set, DataSet
from qcodes.dataset.data_export import reshape_2D_data
from qcodes.dataset.experiment_container import load_by_id, load_experiment
from qcodes.dataset.data_export import get_shaped_data_by_runid
import nanotune as nt

LABELS = list(nt.config["core"]["labels"].keys())
default_coord_names = {'voltage': ['voltage_x', 'voltage_y'],
                        'frequency': ['frequency_x', 'frequency_y']}
default_readout_methods = nt.config['core']['readout_methods']
logger = logging.getLogger(__name__)


class Dataset:
    """
    To be used as parent class for dot fits (in the future)
    For now, it is used to load data and to normalise it with
    normalization_constants.

    Loads existing data from a database, e.g. it expects the data (measured and
    saved with qcodes) to exit in the given database.

    """

    def __init__(
        self,
        qc_run_id: int,
        db_name: Optional[str] = None,
        db_folder: Optional[str] = None,
    ) -> None:

        if db_folder is None:
            db_folder = nt.config["db_folder"]
        self.db_folder = db_folder

        if db_name is None:
            self.db_name, _ = nt.get_database()
        else:
            nt.set_database(db_name, db_folder=db_folder)
            self.db_name = db_name

        self.qc_run_id = qc_run_id
        self._snapshot: Dict[str, Any] = {}
        self._nt_metadata: Dict[str, Any] = {}
        self._normalization_constants: Dict[str, List[float]] = {}

        self.exp_id: int
        self.guid: str
        self.qc_parameters: List[qc.Parameter]
        self.label: List[str]
        self.dimensions: Dict[str, int] = {}
        self.readout_methods: Dict[str, str] = {}

        self.raw_data: xr.Dataset = xr.Dataset()
        self.data: xr.Dataset = xr.Dataset()
        self.power_spectrum: xr.Dataset = xr.Dataset()
        self.filtered_data: xr.Dataset = xr.Dataset()

        self.from_qcodes_dataset()
        self.prepare_filtered_data()
        self.compute_power_spectrum()

    @property
    def snapshot(self) -> Dict[str, Any]:
        """"""
        return self._snapshot

    @property
    def nt_metadata(self) -> Dict[str, Any]:
        """"""
        return self._nt_metadata

    @property
    def normalization_constants(self) -> Dict[str, List[float]]:
        """"""
        return self._normalization_constants

    @property
    def features(self) -> Dict[str, Dict[str, Any]]:
        """"""
        return self._nt_metadata["features"]

    def get_plot_label(self,
                       readout_method: str,
                       axis: int,
                       power_spect: bool = False) -> str:
        """ """
        curr_dim = self.dimensions[readout_method]
        assert curr_dim >= axis
        data = self.data[readout_method]

        if axis == 0 or (axis == 1 and curr_dim > 1):
            if power_spect:
                lbl = default_coord_names['frequency'][axis].replace('_', ' ')
            else:
                lbl = data[default_coord_names['voltage'][axis]].attrs['label']
        else:
            if power_spect:
                lbl = f'power spectrum {data.name}'.replace('_', ' ')
            else:
                ut = data.unit
                lbl = f'{data.name} [{ut}]'.replace('_', ' ')

        return lbl

    def from_qcodes_dataset(self):
        """ Load data from qcodes dataset """
        # qc_dataset = load_by_id(self.qc_run_id)
        qc_dataset = qc.load_by_run_spec(captured_run_id=self.qc_run_id)
        self.exp_id = qc_dataset.exp_id
        self.guid = qc_dataset.guid
        self.qc_parameters = qc_dataset.get_parameters()

        self.raw_data = qc_dataset.to_xarray_dataset()

        self._load_metadata_from_qcodes(qc_dataset)
        self._prep_qcodes_data()

    def _load_metadata_from_qcodes(self, qc_dataset: DataSet):
        self._normalization_constants = {
                key: [0.0, 1.0] for key in ["dc_current", "rf", "dc_sensor"]
            }
        self._snapshot = {}
        self._nt_metadata = {}

        try:
            self._snapshot = json.loads(qc_dataset.get_metadata("snapshot"))
        except (RuntimeError, TypeError) as e:
            pass
        try:
            self._nt_metadata = json.loads(qc_dataset.get_metadata(nt.meta_tag))
        except (RuntimeError, TypeError) as e:
            pass

        try:
            nm = self._nt_metadata["normalization_constants"]
            self._normalization_constants.update(nm)
        except KeyError:
            pass

        try:
            self.device_name = self._nt_metadata["device_name"]
        except KeyError:
            logger.warning("No device name specified.")
            self.device_name = "noname_device"

        try:
            self.readout_methods = self._nt_metadata["readout_methods"]
        except KeyError:
            logger.warning("No readout method specified.")
            self.readout_methods = {}
        if (not isinstance(self.readout_methods, dict) or
            not self.readout_methods):
            if isinstance(self.readout_methods, str):
                methods = [self.readout_methods]
            elif isinstance(self.readout_methods, list):
                methods = self.readout_methods
            else:
                # we assume the default order of readout methods if nothing
                # else is specified
                methods = default_readout_methods[:len(self.raw_data)]
            read_params = [str(it) for it in list(self.raw_data.data_vars)]
            self.readout_methods = dict(zip(methods, read_params))

        quality = qc_dataset.get_metadata("good")
        if quality is not None:
            self.quality = int(quality)
        else:
            self.quality = None

        self.label = []
        for label in LABELS:
            if label != "good":
                lbl = qc_dataset.get_metadata(label)
                if lbl is None:
                    lbl = 0
                if int(lbl) == 1:
                    self.label.append(label)

    def _prep_qcodes_data(self, rename: bool = True):
        """ normalize data and rename labels """
        self.data = self.raw_data.interpolate_na()
        if rename:
            self._rename_xarray_variables()

        for r_meth, r_param in self.readout_methods.items():
            self.dimensions[r_meth] = len(self.raw_data[r_param].dims)
            nrm_dt = self._normalize_data(self.raw_data[r_param].values, r_meth)
            self.data[r_meth].values = nrm_dt

            for vi, vr in enumerate(self.raw_data[r_param].depends_on):
                coord_name = default_coord_names['voltage'][vi]
                lbl = self._nt_label(r_param, vr)
                self.data[r_meth][coord_name].attrs['label'] = lbl

    def _nt_label(self, readout_paramter, var) -> str:
        lbl = self.raw_data[readout_paramter][var].attrs['label']
        unit = self.raw_data[readout_paramter][var].attrs['unit']
        return f'{lbl} [{unit}]'

    def _rename_xarray_variables(self):
        new_names = {old: new for new, old in self.readout_methods.items()}
        new_names_all = copy.deepcopy(new_names)
        for old_n in new_names.keys():
            for c_i, old_crd in enumerate(self.raw_data[old_n].depends_on):
                new_names_all[old_crd] = default_coord_names['voltage'][c_i]

        self.data = self.data.rename(new_names_all)

    def _normalize_data(self, signal: np.ndarray, signal_type) -> np.ndarray:
        """"""
        minv = self.normalization_constants[signal_type][0]
        maxv = self.normalization_constants[signal_type][1]

        normalized_sig = (signal - minv) / (maxv - minv)
        if np.max(normalized_sig) > 1 or np.min(normalized_sig) < 0:
            msg = (
                "Dataset {}: ".format(self.qc_run_id),
                "Wrong normalization constant",
            )
            logger.warning(msg)
        return normalized_sig

    def prepare_filtered_data(self):
        """"""
        self.filtered_data = self.data.copy(deep=True)
        for read_meth in self.readout_methods:
            smooth = gaussian_filter(self.filtered_data[read_meth].values,
                                     sigma=2)
            self.filtered_data[read_meth].values = smooth


    def compute_1D_power_spectrum(
        self, readout_method: str,
    ) -> xr.DataArray:
        """"""
        c_name = default_coord_names['voltage'][0]
        voltage_x = self.data[readout_method][c_name].values

        xv = np.unique(voltage_x)
        signal = self.data[readout_method].values.copy()
        signal = sg.detrend(signal, axis=0)

        frequencies_res = fp.fft(signal)
        frequencies_res = np.abs(fp.fftshift(frequencies_res))**2

        fx = fp.fftshift(fp.fftfreq(frequencies_res.shape[0], d=xv[1] - xv[0]))

        freq_xar = xr.DataArray(
                    frequencies_res,
                    coords=[(default_coord_names['frequency'][0], fx)],
                )
        return freq_xar

    def compute_2D_power_spectrum(
        self,
        readout_method: str,
    ) -> xr.DataArray:
        """"""
        c_name_x = default_coord_names['voltage'][0]
        c_name_y = default_coord_names['voltage'][1]
        voltage_x = self.data[readout_method][c_name_x].values
        voltage_y = self.data[readout_method][c_name_y].values
        signal = self.data[readout_method].values.copy()

        xv = np.unique(voltage_x)
        yv = np.unique(voltage_y)
        signal = signal.copy()

        signal = sg.detrend(signal, axis=0)
        signal = sg.detrend(signal, axis=1)

        frequencies_res = fp.fft2(signal)
        frequencies_res = np.abs(fp.fftshift(frequencies_res))**2

        fx_1d = fp.fftshift(fp.fftfreq(frequencies_res.shape[0],
                                       d=xv[1] - xv[0]))
        fy_1d = fp.fftshift(fp.fftfreq(frequencies_res.shape[1],
                                       d=yv[1] - yv[0]))

        freq_xar = xr.DataArray(
                frequencies_res,
                coords=[(default_coord_names['frequency'][0], fx_1d),
                        (default_coord_names['frequency'][1], fy_1d)],
            )

        return freq_xar

    def compute_power_spectrum(self):
        """
        Compute frequencies of the signal. Detrend before that to eliminate DC
        component.
        No high pass filter applied as we do not want to accidentally remove
        information contained in low frequencies. We count on PCA to assess
        which signal/frequencies are important
        """
        self.power_spectrum = xr.Dataset()

        for readout_method in self.readout_methods.keys():
            if self.dimensions[readout_method] == 1:
                freq_xar = self.compute_1D_power_spectrum(readout_method)
            elif self.dimensions[readout_method] == 2:
                freq_xar = self.compute_2D_power_spectrum(readout_method)
            else:
                raise NotImplementedError

            self.power_spectrum[readout_method] = freq_xar


