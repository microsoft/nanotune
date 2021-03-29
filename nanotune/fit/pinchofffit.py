import os
import json
import copy
import logging

import numpy as np
import scipy as sc
from scipy.signal import argrelextrema

from typing import Optional, Dict, List, Tuple, Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linalg as lg

# from matplotlib2tikz import save as tikz_save

from scipy.optimize import least_squares
from scipy.stats import norm

# from qcodes.self.experiment_container import load_by_id

import nanotune as nt
from nanotune.fit.datafit import DataFit
from nanotune.utils import format_axes
from nanotune.data.plotting import (default_plot_params,
                                    colors_dict, plot_params_type)

logger = logging.getLogger(__name__)
AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]


class PinchoffFit(DataFit):
    """"""

    def __init__(
        self,
        qc_run_id: int,
        db_name: str,
        db_folder: Optional[str] = None,
        gradient_percentile: float = 25,
        get_transition_from_fit: bool = False,
    ) -> None:
        if db_folder is None:
            db_folder = nt.config["db_folder"]

        DataFit.__init__(
            self,
            qc_run_id,
            db_name,
            db_folder=db_folder,
        )

        self.gradient_percentile = gradient_percentile

        self.low_signal: Dict[str, float] = {}
        self.high_signal: Dict[str, float] = {}
        self.transition_voltage: Dict[str, float] = {}
        self.transition_signal: Dict[str, float] = {}

        self.low_signal_index: Dict[str, float] = {}
        self.high_signal_index: Dict[str, float] = {}
        self.transition_signal_index: Dict[str, float] = {}

        self.get_transition_from_fit = get_transition_from_fit

        n_points = self.data['dc_current']['voltage_x'].shape[0]
        self.normalized_voltage = np.linspace(0, 1, n_points)

    @property
    def next_actions(self) -> Dict[str, List[str]]:
        """"""
        if not self._low_signal:
            self.compute_transition_interval()
        self._next_actions = {}
        for read_meth in self.readout_methods:
            self._next_actions[read_meth] = []

            if self.high_signal[read_meth] < 0.8:
                self._next_actions[read_meth].append("x more negative")
            if self.low_signal[read_meth] > 0.2:
                self._next_actions[read_meth].append("x more positive")

        return self._next_actions

    def find_fit(self) -> None:
        """"""
        for r_meth in self.readout_methods:
            (bounds,
             initial_guess) = self.compute_initial_guess(readout_method=r_meth)
            smooth_signal = self.filtered_data[r_meth].values

            def err_func(p: List[float]) -> np.ndarray:
                err = self.fit_fct(self.normalized_voltage, p)
                err = (err - smooth_signal) / lg.norm(smooth_signal)
                return err

            sub_res = least_squares(
                err_func,
                initial_guess,
                method="trf",
                loss="cauchy",
                verbose=0,
                bounds=bounds,
                gtol=1e-15,
                ftol=1e-15,
                xtol=1e-15,
            )

            self._retain_fit_result(sub_res, r_meth)

        self.compute_transition_interval()
        self.compute_transition_voltage()

        self._retain_transition_features()
        self.save_features()

    def _retain_fit_result(self, subres, read_meth):
        self._features[read_meth] = {}
        self._features[read_meth]["amplitude"] = subres.x[0]
        self._features[read_meth]["slope"] = subres.x[1]
        self._features[read_meth]["offset"] = subres.x[2]
        self._features[read_meth]["residuals"] = lg.norm(subres.fun)

    def _retain_transition_features(self) -> None:
        for read_meth in self.readout_methods:
            v_x = self.data[read_meth]['voltage_x'].values
            low_idx = self.low_signal_index[read_meth]
            high_idx = self.high_signal_index[read_meth]
            trans_v = self.transition_voltage[read_meth]
            trans_s = self.transition_signal[read_meth]

            self._features[read_meth]["low_voltage"] = v_x[low_idx]
            self._features[read_meth]["high_voltage"] = v_x[high_idx]
            self._features[read_meth]["low_signal"] = self.low_signal[read_meth]
            self._features[read_meth]["high_signal"] = self.high_signal[read_meth]
            self._features[read_meth]["transition_voltage"] = trans_v
            self._features[read_meth]["transition_signal"] = trans_s
            sign = self.data[read_meth].values
            self._features[read_meth]["max_signal"] = np.max(sign)
            self._features[read_meth]["min_signal"] = np.min(sign)


    def compute_transition_interval(self) -> None:
        """
        Using the signal gradient to determine where the transition from high
         to low voltage occurs
        """
        for read_meth in self.readout_methods:
            if not self.get_transition_from_fit:
                temp_sig = self.filtered_data[read_meth].values
                temp_v = self.filtered_data[read_meth]['voltage_x'].values
            else:
                fit_feat = [
                    self.features[read_meth]["amplitude"],
                    self.features[read_meth]["slope"],
                    self.features[read_meth]["offset"],
                ]
                temp_sig = self.fit_fct(self.normalized_voltage, fit_feat)
                temp_v = self.normalized_voltage

            gradient = np.gradient(temp_sig, temp_v)
            max_gradient = np.max(gradient)
            max_gradient = self.gradient_percentile * max_gradient / 100
            logger.debug(f"max_gradient = {max_gradient}")

            range_idx = np.where(gradient >= max_gradient)[0]
            logger.debug(f"Valid range indexes: {range_idx}")

            if range_idx[0].size > 0:
                low_idx = np.where(temp_sig == min(temp_sig[range_idx]))[0][0]
                high_idx = np.where(temp_sig == max(temp_sig[range_idx]))[0][0]
            else:
                low_idx = np.where(temp_sig == min(temp_sig))[0][0]
                high_idx = np.where(temp_sig == max(temp_sig))[0][0]
                logger.warning("No good valid range found.")

            self.low_signal[read_meth] = temp_sig[low_idx]
            self.high_signal[read_meth] = temp_sig[high_idx]

            self.low_signal_index[read_meth] = low_idx
            self.high_signal_index[read_meth] = high_idx

    def compute_transition_voltage(self) -> None:
        """"""
        for read_meth in self.readout_methods:
            if not self.get_transition_from_fit:
                temp_sig = self.filtered_data[read_meth].values
                temp_v = self.filtered_data[read_meth]['voltage_x'].values
            else:
                fit_feat = [
                    self.features[read_meth]["amplitude"],
                    self.features[read_meth]["slope"],
                    self.features[read_meth]["offset"],
                ]
                temp_sig = self.fit_fct(self.normalized_voltage, fit_feat)
                temp_v = self.normalized_voltage

            fit_gradient = np.gradient(temp_sig, temp_v)
            relevant_idx = np.argmax(fit_gradient)
            self.transition_signal_index[read_meth] = relevant_idx

            self.transition_signal[read_meth] = temp_sig[relevant_idx]
            self.transition_voltage[read_meth] = temp_v[relevant_idx]

    def compute_initial_guess(
        self,
        readout_method: str = 'dc_current',
    ) -> Tuple[Tuple[List[float], List[float]], List[float]]:
        """"""
        signal = self.data[readout_method].values
        amplitude_init = np.max(signal) - np.min(signal)
        amplitude_min = 0
        amplitude_max = amplitude_init

        slope_init = 1
        slope_init /= amplitude_init
        slope_min = 0
        slope_max = np.inf

        # voltage_offset is the horizontal offset: not important
        # We do not care where in voltage space the transition happens
        # but we need to shift the fit horizontally to make it overlap
        # with the data
        voltage_offset_init = (
            np.max(self.normalized_voltage) - np.min(self.normalized_voltage)
        ) / 2
        voltage_offset_min = -np.inf
        voltage_offset_max = np.inf

        min_bounds = [amplitude_min, slope_min, voltage_offset_min]
        max_bounds = [amplitude_max, slope_max, voltage_offset_max]

        bounds = (min_bounds, max_bounds)
        initial_guess = [amplitude_init, slope_init, voltage_offset_init]

        return bounds, initial_guess

    def fit_fct(self, v: np.ndarray, params: List[float]) -> np.ndarray:
        """
        Function we use to fit pinch off curves.

        x: input vector, usually gate voltage in V
        a: parameter referred to as 'amplitude'
        b: parameter referred to as 'slope'
        c: a (kind of) shift
        """
        fit = 1 + np.tanh(params[1] * v + params[2])
        return params[0] * fit

    def _retain_fit_result(self, subres, read_meth):
        self._features[read_meth] = {}
        self._features[read_meth]["amplitude"] = subres.x[0]
        self._features[read_meth]["slope"] = subres.x[1]
        self._features[read_meth]["offset"] = subres.x[2]
        self._features[read_meth]["residuals"] = lg.norm(subres.fun)

    def _retain_transition_features(self) -> None:
        for read_meth in self.readout_methods:
            v_x = self.data[read_meth]['voltage_x'].values
            low_idx = self._low_signal_index[read_meth]
            high_idx = self._high_signal_index[read_meth]
            trans_v = self._transition_voltage[read_meth]
            trans_s = self._transition_signal[read_meth]

            self._features[read_meth]["low_voltage"] = v_x[low_idx]
            self._features[read_meth]["high_voltage"] = v_x[high_idx]
            self._features[read_meth]["low_signal"] = self._low_signal[read_meth]
            self._features[read_meth]["high_signal"] = self._high_signal[read_meth]
            self._features[read_meth]["transition_voltage"] = trans_v
            self._features[read_meth]["transition_signal"] = trans_s
            sign = self.data[read_meth].values
            self._features[read_meth]["max_signal"] = np.max(sign)
            self._features[read_meth]["min_signal"] = np.min(sign)

    def plot_fit(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        colorbar: Optional[matplotlib.colorbar.Colorbar] = None,
        save_figures: bool = True,
        plot_gradient: Optional[bool] = True,
        plot_smooth: Optional[bool] = True,
        filename: Optional[str] = None,
        file_location: Optional[str] = None,
        plot_params: Optional[plot_params_type] = None,
    ) -> AxesTuple:
        """"""
        if plot_params is None:
            plot_params = default_plot_params
        if not self.high_signal_index:
            self.find_fit()
        matplotlib.rcParams.update(plot_params)
        fig_title = f"Pinchoff fit {self.guid}"

        arrowprops=dict([
                    ('arrowstyle', "->"),
                    ('color', plot_params["axes.edgecolor"]),
                    ('shrinkA', 5),
                    ('shrinkB', 5),
                    ('patchA', None),
                    ('patchB', None),
                    ('connectionstyle', "arc3,rad=0"),
        ])
        bbox=dict([
                    ('boxstyle', "round,pad=0.4"),
                    ('fc', "white"),
                    ('ec', plot_params["axes.edgecolor"]),
                    ('lw', plot_params["axes.linewidth"]),
        ])

        if ax is None:
            fig_size = copy.deepcopy(plot_params["figure.figsize"])
            fig_size[1] *= len(self.readout_methods) * 0.8  # type: ignore
            fig, ax = plt.subplots(len(self.readout_methods),
                                   1, squeeze=False, figsize=fig_size)

        for r_i, read_meth in enumerate(self.readout_methods):
            voltage = self.data[read_meth]['voltage_x'].values
            signal = self.data[read_meth].values

            ax[r_i, 0].plot(voltage, signal, label="signal", zorder=6)
            ax[r_i, 0].set_xlabel(self.get_plot_label(read_meth, 0))
            ax[r_i, 0].set_ylabel(self.get_plot_label(read_meth, 1))
            ax[r_i, 0].set_title(str(fig_title))

            H_v = voltage[self.high_signal_index[read_meth]]
            H_s = signal[self.high_signal_index[read_meth]]
            ax[r_i, 0].annotate(
                "H",
                xy=(H_v, H_s),
                xytext=(0, -40),
                textcoords="offset points",
                ha="right",
                va="bottom",
                bbox=bbox,
                arrowprops=arrowprops,
            )
            L_v = voltage[self.low_signal_index[read_meth]]
            L_s = signal[self.low_signal_index[read_meth]]
            ax[r_i, 0].annotate(
                "L",
                xy=(L_v, L_s),
                xytext=(0, 20),
                textcoords="offset points",
                ha="right",
                va="bottom",
                bbox=bbox,
                arrowprops=arrowprops,
            )
            T_v = voltage[self.transition_signal_index[read_meth]]
            T_s = signal[self.transition_signal_index[read_meth]]
            ax[r_i, 0].annotate(
                "T",
                xy=(T_v, T_s),
                xytext=(0, 20),
                textcoords="offset points",
                ha="right",
                va="bottom",
                bbox=bbox,
                arrowprops=arrowprops,
            )

            if plot_smooth:
                smooth = self.filtered_data[read_meth].values
                ax[r_i, 0].plot(
                    voltage, smooth, label="smooth", zorder=2
                )

            if plot_gradient:
                gradient = np.gradient(signal, voltage)
                gradient /= np.max(gradient)

                ax[r_i, 0].plot(voltage, gradient, label="gradient", zorder=0)

            fit_feat = [
                self.features[read_meth]["amplitude"],
                self.features[read_meth]["slope"],
                self.features[read_meth]["offset"],
            ]
            fit = self.fit_fct(self.normalized_voltage, fit_feat)
            ax[r_i, 0].plot(voltage, fit, label="fit", zorder=4)
            ax[r_i, 0].legend(loc="lower right", bbox_to_anchor=(1, 0.1))

            divider = make_axes_locatable(ax[r_i, 0])
            cbar_ax = divider.append_axes("right", size="5%", pad=0.06)
            cbar_ax.set_facecolor("none")
            for caxis in ["top", "bottom", "left", "right"]:
                cbar_ax.spines[caxis].set_linewidth(0)
            cbar_ax.set_xticks([])
            cbar_ax.set_yticks([])

        fig.tight_layout()

        if save_figures:
            if file_location is None:
                file_location = os.path.join(
                    nt.config["db_folder"], "tuning_results", self.device_name
                )
            if not os.path.exists(file_location):
                os.makedirs(file_location)

            if filename is None:
                filename = f"pinchofffit_{self.guid}"
            else:
                filename = os.path.splitext(filename)[0]

            path = os.path.join(file_location, filename + ".png")
            plt.savefig(path, format="png", dpi=600, bbox_inches="tight")

        return ax, None

    def plot_features(
        self,
        save_figures: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        highlight_color: Optional[str] = "indianred",
        fill_color: Optional[str] = "indianred",
        fill_hatch: Optional[str] = None,
        filename: Optional[str] = None,
        file_location: Optional[str] = None,
        plot_params: Optional[plot_params_type] = None,
    ) -> AxesTuple:
        """"""
        if plot_params is None:
            plot_params = default_plot_params
        matplotlib.rcParams.update(plot_params)

        if not self.high_signal_index:
            self.find_fit()

        fig_title = f"Pinchoff features {self.guid}"

        if ax is None:
            fig_size = copy.deepcopy(plot_params["figure.figsize"])
            fig_size[1] *= len(self.readout_methods) * 0.8  # type: ignore
            fig, ax = plt.subplots(len(self.readout_methods), 1,
                                   squeeze=False, figsize=fig_size)

        color = highlight_color
        pad = 0.02

        for r_i, read_meth in enumerate(self.readout_methods.keys()):
            voltage = self.data[read_meth]['voltage_x'].values
            signal = self.data[read_meth].values

            ax[r_i, 0].plot(voltage, signal, linestyle="-",
                            label="signal", zorder=10)
            ax[r_i, 0].set_xlabel(self.get_plot_label(read_meth, 0))
            ax[r_i, 0].set_ylabel(self.get_plot_label(read_meth, 1))
            ax[r_i, 0].set_title(fig_title)

            fit_feat = [
                self.features[read_meth]["amplitude"],
                self.features[read_meth]["slope"],
                self.features[read_meth]["offset"],
            ]

            ax[r_i, 0].plot(
                voltage,
                self.fit_fct(self.normalized_voltage, fit_feat),
                linestyle="-",
                label="fit",
                zorder=6,
            )

            ax[r_i, 0].fill_between(
                voltage,
                signal,
                self.fit_fct(self.normalized_voltage, fit_feat),
                label="residuals",
                color=fill_color,
                hatch=fill_hatch,
            )  # '0.75')

            pad_n = 2
            T_idx = self.transition_signal_index[read_meth]
            H_idx = self.high_signal_index[read_meth]
            L_idx = self.low_signal_index[read_meth]

            ax[r_i, 0].vlines(
                x=voltage[T_idx + pad_n] + 1 * pad,
                ymin=signal[T_idx - pad_n],
                ymax=signal[T_idx + pad_n],
                linestyles="dashed",
                color=color,
            )
            ax[r_i, 0].hlines(
                y=signal[T_idx - pad_n],
                xmin=voltage[T_idx - pad_n] + 1 * pad,
                xmax=voltage[T_idx + pad_n] + 1 * pad,
                linestyles="dashed",
                color=color,
            )
            ax[r_i, 0].plot(
                (voltage[T_idx - pad_n] + 1 * pad, voltage[T_idx + pad_n] + 1 * pad),
                (signal[T_idx - pad_n], signal[T_idx + pad_n]),
                color=color,
                linestyle="dashed",
            )
            ax[r_i, 0].text(
                voltage[T_idx + pad_n] + 2 * pad, signal[T_idx] - pad, "slope"
            )

            ax[r_i, 0].hlines(
                y=signal[T_idx],
                xmin=np.min(voltage),
                xmax=voltage[T_idx],
                linestyles="dashed",
                color=color,
            )
            ax[r_i, 0].text(
                np.min(voltage),
                signal[T_idx] + pad,
                "transition",
                fontsize=plot_params["font.size"],
            )

            ax[r_i, 0].hlines(
                y=signal[H_idx],
                xmin=np.min(voltage),
                xmax=voltage[H_idx],
                linestyles="dashed",
                color=color,
            )
            ax[r_i, 0].text(np.min(voltage), signal[H_idx] + pad, "high")

            ax[r_i, 0].hlines(
                y=signal[L_idx],
                xmin=np.min(voltage),
                xmax=voltage[L_idx],
                linestyles="dashed",
                zorder=10,
                color=color,
            )
            ax[r_i, 0].text(np.min(voltage), signal[L_idx] + pad, "low")

            ax[r_i, 0].legend(loc="lower right", bbox_to_anchor=(1, 0))

            divider = make_axes_locatable(ax[r_i, 0])
            cbar_ax = divider.append_axes("right", size="5%", pad=0.06)
            cbar_ax.set_facecolor("none")
            for caxis in ["top", "bottom", "left", "right"]:
                cbar_ax.spines[caxis].set_linewidth(0)
            cbar_ax.set_xticks([])
            cbar_ax.set_yticks([])

            ax[r_i, 0].set_aspect("auto")

        fig.tight_layout()
        if save_figures:
            if file_location is None:
                file_location = os.path.join(
                    nt.config["db_folder"], "tuning_results", self.device_name
                )
            if not os.path.exists(file_location):
                os.makedirs(file_location)

            if filename is None:
                filename = f"pinchoff_features_{self.guid}"
            else:
                filename = os.path.splitext(filename)[0]

            path = os.path.join(file_location, filename + ".png")
            plt.savefig(path, format="png", dpi=600, bbox_inches="tight")

        return ax, None
