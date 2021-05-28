import os
import logging
import copy

import numpy as np
import scipy as sc

from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt

import nanotune as nt
from nanotune.fit.datafit import DataFit
from nanotune.utils import format_axes
from nanotune.data.plotting import default_plot_params, colors_dict, plot_params_type

logger = logging.getLogger(__name__)
AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]


class CoulombOscillationFit(DataFit):
    """"""

    def __init__(
        self,
        qc_run_id: int,
        db_name: str,
        db_folder: Optional[str] = None,
        relative_height_threshold: float = 0.5,
        sigma_dV: float = 0.005,
    ) -> None:

        if db_folder is None:
            db_folder = nt.config["db_folder"]

        DataFit.__init__(
            self,
            qc_run_id,
            db_name,
            db_folder=db_folder,
        )

        self.relative_height_threshold = relative_height_threshold
        self.sigma_dV = sigma_dV

        self.peak_indx: Dict[str, List[int]] = {}
        self.peak_distances: Dict[str, List[float]] = {}

    @property
    def range_update_directives(self) -> List[str]:
        """"""
        raise NotImplementedError

    def find_fit(self) -> None:
        """
        Find peaks and extract distances between them (in voltage space)
        """
        self.peak_indx = self.find_peaks()
        self.peak_distances = self.calculate_peak_distances(self.peak_indx)
        self.peak_locations = self.get_peak_locations()

        self._retain_fit_result()
        self.save_features()

    def _retain_fit_result(self):
        self._features = {}
        for read_meth in self.readout_methods.keys():
            self._features[read_meth] = {}
            self._features[read_meth]["peak_indx"] = self.peak_indx[read_meth]
            temp = self.peak_locations[read_meth]
            self._features[read_meth]["peak_locations"] = temp
            temp = self.peak_distances[read_meth]
            self._features[read_meth]["peak_distances"] = temp

    def calculate_voltage_distances(self) -> Dict[str, float]:
        """
        Get voltage spacing between Coulomb peaks
        """
        voltage_distances = {}
        for read_meth in self.readout_methods.keys():
            voltage_distances[read_meth] = np.max(self.peak_distances[read_meth])
        return voltage_distances

    def get_peak_locations(self) -> Dict[str, List[float]]:
        peak_locations = {}
        for read_meth in self.readout_methods.keys():
            v_x = self.data[read_meth].voltage_x.values
            peak_idx = self.peak_indx[read_meth]
            peak_locations[read_meth] = v_x[peak_idx].tolist()

        return peak_locations

    def find_peaks(
        self,
        absolute_height_threshold: Optional[float] = None,
        minimal_index_distance: int = 3,
    ) -> Dict[str, List[int]]:
        """
        wrapper around scipy.signal.peaks

        the threshold calculated with:
        height = height_threshold * np.max(self.signal)
        """
        peaks = {}
        for read_meth in self.readout_methods.keys():
            if absolute_height_threshold is None:
                absolute_height_threshold = self.relative_height_threshold
                smooth_curr = self.filtered_data[read_meth].values
                absolute_height_threshold *= np.max(smooth_curr)
            self.absolute_height_threshold = absolute_height_threshold

            found_peaks, _ = sc.signal.find_peaks(
                self.filtered_data[read_meth].values,
                height=[self.absolute_height_threshold, None],
                distance=minimal_index_distance,
            )
            peaks[read_meth] = found_peaks.tolist()
        return peaks

    def calculate_peak_distances(
        self,
        peak_indx: Dict[str, List[int]],
    ) -> Dict[str, List[float]]:
        """"""
        peak_distances: Dict[str, List[float]] = {}
        for read_meth in self.readout_methods.keys():
            voltage = self.data[read_meth].voltage_x.values
            peak_distances[read_meth] = []
            peaks = peak_indx[read_meth]
            if len(peaks) > 1:
                for ip in range(len(peaks) - 1):
                    peak = peaks[ip]
                    next_peak = peaks[ip + 1]
                    d = voltage[peak] - voltage[next_peak]
                    peak_distances[read_meth].append(abs(d))

        return peak_distances

    def plot_fit(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        save_figures: bool = True,
        filename: Optional[str] = None,
        file_location: Optional[str] = None,
        plot_params: Optional[plot_params_type] = None,
    ) -> AxesTuple:
        """"""
        if plot_params is None:
            plot_params = default_plot_params
        matplotlib.rcParams.update(plot_params)
        fig_title = f"Coulomboscillation fit {self.guid}"

        if not self.peak_indx:
            self.find_fit()

        if ax is None:
            fig_size = copy.deepcopy(plot_params["figure.figsize"])
            fig_size[1] *= len(self.data) * 0.8  # type: ignore
            fig, ax = plt.subplots(len(self.data), 1, squeeze=False, figsize=fig_size)

        for r_i, read_meth in enumerate(self.readout_methods.keys()):
            voltage = self.data[read_meth]["voltage_x"].values
            signal = self.data[read_meth].values
            smooth_sig = self.filtered_data[read_meth].values

            ax[r_i, 0].plot(
                voltage,
                signal,
                color=colors_dict["blue"],
                label="signal",
                zorder=6,
            )
            ax[r_i, 0].set_xlabel(self.get_plot_label(read_meth, 0))
            ax[r_i, 0].set_ylabel(self.get_plot_label(read_meth, 1))
            ax[r_i, 0].set_title(fig_title)

            ax[r_i, 0].plot(
                voltage,
                smooth_sig,
                color=colors_dict["orange"],
                label="smooth",
                zorder=2,
            )
            ax[r_i, 0].plot(
                voltage[self.peak_indx[read_meth]],
                smooth_sig[self.peak_indx[read_meth]],
                "x",
                color=colors_dict["teal"],
                label="peaks",
            )
            ax[r_i, 0].vlines(
                x=voltage[self.peak_indx[read_meth]],
                ymin=0,
                ymax=smooth_sig[self.peak_indx[read_meth]],
                color=colors_dict["teal"],
                linestyles="dashed",
            )

            height = self.absolute_height_threshold
            ax[r_i, 0].plot(
                voltage,
                np.zeros_like(smooth_sig) + height,
                "--",
                color="gray",
                label="threshold",
            )

            ax[r_i, 0].legend(
                loc="upper right",
                bbox_to_anchor=(1, 1),
                frameon=False,
            )
            ax[r_i, 0].set_ylabel("normalized signal")

            ax[r_i, 0].set_aspect("auto")
            ax[r_i, 0].figure.tight_layout()

        fig.tight_layout()

        if save_figures:
            if file_location is None:
                file_location = os.path.join(
                    nt.config["db_folder"], "tuning_results", self.device_name
                )
            if not os.path.exists(file_location):
                os.makedirs(file_location)

            if filename is None:
                filename = f"coulomboscillationfit_{self.guid}"
            else:
                filename = os.path.splitext(filename)[0]

            path = os.path.join(file_location, filename + ".png")
            plt.savefig(path, format="png", dpi=600, bbox_inches="tight")

        return ax, None
