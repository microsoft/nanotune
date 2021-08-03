import copy
import json
import logging
import math
import os
from itertools import combinations
from math import floor
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy.typing as npt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qcodes as qc
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qcodes.dataset.measurements import Measurement
from scipy.ndimage import measurements as scm
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure

import nanotune as nt
from nanotune.data.dataset import default_coord_names
from nanotune.data.plotting import default_plot_params
from nanotune.fit.datafit import DataFit

logger = logging.getLogger(__name__)

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
default_dot_fit_parameters: Dict[str, Dict[str, Union[int, float]]] = {
    "transport": {
        "noise_level": 0.02,
        "binary_neighborhood": 1,
        "distance_threshold": 0.05,
    },
    "sensing": {
        "noise_level": 0.3,
        "binary_neighborhood": 2,
        "distance_threshold": 0.05,
    },
}


class DotFit(DataFit):
    def __init__(
        self,
        qc_run_id: int,
        db_name: str,
        db_folder: Optional[str] = None,
        segment_size: float = 0.05,
        noise_floor: float = 0.004,
        dot_signal_threshold: float = 0.1,
        fit_parameters: Optional[Dict[str, Dict[str, Union[int, float]]]] = None,
        **kwargs,
    ) -> None:

        if db_folder is None:
            db_folder = nt.config["db_folder"]
        if fit_parameters is None:
            fit_parameters = default_dot_fit_parameters

        DataFit.__init__(
            self,
            qc_run_id,
            db_name,
            db_folder=db_folder,
            **kwargs,
        )

        self.signal_thresholds = [noise_floor, dot_signal_threshold]
        self.segment_size = segment_size
        self.fit_parameters = fit_parameters
        self.segmented_data: List[xr.Dataset] = []

        self.triple_points: Optional[Dict[str, List[float]]] = None

    @property
    def range_update_directives(self) -> List[str]:
        """
        signal_type: If more than one signal type (i.e dc, sensing or rf)
        have been measured, select which one to use to perform the edge
        analysis on. Use indexing and the order in which these signals have been
        measured.
        If the data has been segmented, the entire measurement is used to
        determine next voltage range updates.
        """
        if not self._range_update_directives:

            for read_meth in self.readout_methods:
                left_vertical = self.get_edge("left vertical", read_meth)
                bottom_horizontal = self.get_edge("bottom horizontal", read_meth)
                right_vertical = self.get_edge("right vertical", read_meth)
                top_horizontal = self.get_edge("top horizontal", read_meth)

                if np.max(left_vertical) > self.signal_thresholds[1]:
                    self._range_update_directives.append("x more negative")

                if np.max(bottom_horizontal) > self.signal_thresholds[1]:
                    self._range_update_directives.append("y more negative")

                if np.max(right_vertical) < self.signal_thresholds[0]:
                    self._range_update_directives.append("x more positive")

                if np.max(top_horizontal) < self.signal_thresholds[0]:
                    self._range_update_directives.append("y more positive")

        return self._range_update_directives

    def find_fit(self) -> None:
        """
        Until capacitance extraction is not implemented no features are
        returned.
        """
        self.triple_points = self.get_triple_point_distances()
        for read_meth in self.readout_methods:
            self._features[read_meth] = {}
            tps = self.triple_points[read_meth]
            self._features[read_meth]["triple_points"] = list(tps)

    def prepare_segmented_data(
        self,
        use_raw_data: bool = False,
    ) -> None:
        """"""
        if use_raw_data:
            data = self.raw_data
            coord_names = [str(it) for it in list(self.raw_data.coords)]
            readout_params = list(self.readout_methods.values())
        else:
            data = self.data
            coord_names = default_coord_names["voltage"]
            readout_params = list(self.readout_methods.keys())

        self.segmented_data = []

        for read_meth in readout_params:
            orig_v_x = data[read_meth][coord_names[0]].values
            orig_v_y = data[read_meth][coord_names[1]].values
            signal = data[read_meth].values
            orig_shape_x, orig_shape_y = signal.shape

            vx_span = abs(orig_v_x[0] - orig_v_x[-1])
            # round to 8 digits to avoid e.g 0.0999999999/0.05
            # to be floored to 1
            vx_span = round(vx_span, 8)
            n_x = int(floor(vx_span / self.segment_size))

            vy_span = abs(orig_v_y[0] - orig_v_y[-1])
            vy_span = round(vy_span, 8)
            n_y = int(floor(vy_span / self.segment_size))

            if n_x >= orig_shape_x / 10 or n_y >= orig_shape_x / 10:
                logger.warning(f"Dotfit {self.guid}: Mesh resolution too low.")

            n_mesh = [np.max([1, n_x]), np.max([1, n_y])]
            if not self.segmented_data:
                empty = [xr.Dataset() for i in range(np.prod(n_mesh))]
                self.segmented_data = empty
            dind_x, indx_remain = divmod(orig_shape_x, n_mesh[0])
            dind_y, indy_remain = divmod(orig_shape_y, n_mesh[1])
            n_total = 0
            for n1 in range(0, n_mesh[0]):
                for n2 in range(0, n_mesh[1]):
                    n_total += 1
                    i1_start = n1 * dind_x
                    i1_stop = (n1 + 1) * dind_x
                    if n1 == n_mesh[0] - 1:
                        i1_stop += indx_remain

                    i2_start = n2 * dind_y
                    i2_stop = (n2 + 1) * dind_y
                    if n2 == n_mesh[1] - 1:
                        i2_stop += indy_remain

                    segment_x = orig_v_x[i1_start:i1_stop]
                    segment_y = orig_v_y[i2_start:i2_stop]
                    signal_sgmt = signal[i1_start:i1_stop, i2_start:i2_stop]

                    seg_xar = xr.DataArray(
                        signal_sgmt,
                        coords=[
                            (coord_names[0], segment_x),
                            (coord_names[1], segment_y),
                        ],
                    )
                    seg_xrset = xr.Dataset({read_meth: seg_xar})

                    self.segmented_data[n_total - 1].update(seg_xrset)

    def save_segmented_data_return_info(
        self,
        segment_db_name: str,
        segment_db_folder: Optional[str] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Save each mesh in a new dataset in given databases

        Returns:
            dict: nested, with the data ID as key on the first level, readout
                method and `voltage_ranges` on the second.
        """
        if segment_db_folder is None:
            segment_db_folder = nt.config["db_folder"]

        if not self.segmented_data:
            self.prepare_segmented_data(use_raw_data=True)
        if not os.path.isfile(os.path.join(segment_db_folder, segment_db_name)):
            ds = qc.load_by_run_spec(captured_run_id=self.qc_run_id)
            nt.new_database(segment_db_name, db_folder=segment_db_folder)
            qc.new_experiment(f"segmented_{ds.exp_name}", sample_name=ds.sample_name)

        original_params = self.qc_parameters
        segment_info: Dict[int, Dict[str, Any]] = {}

        with nt.switch_database(segment_db_name, segment_db_folder):
            for segment in self.segmented_data:
                meas = Measurement()
                meas.register_custom_parameter(
                    original_params[0].name,
                    label=original_params[0].label,
                    unit=original_params[0].unit,
                    paramtype="array",
                )

                meas.register_custom_parameter(
                    original_params[1].name,
                    label=original_params[1].label,
                    unit=original_params[1].unit,
                    paramtype="array",
                )
                result: List[List[Tuple[str, npt.NDArray[np.float64]]]] = []
                ranges: List[Tuple[float, float]] = []
                m_params = [str(it) for it in list(segment.data_vars)]
                for ip, param_name in enumerate(m_params):
                    coord_names = list(segment.coords)
                    x_crd_name = coord_names[0]
                    y_crd_name = coord_names[1]

                    voltage_x = segment[param_name][x_crd_name].values
                    voltage_y = segment[param_name][y_crd_name].values
                    signal = segment[param_name].values

                    range_x = (np.min(voltage_x), np.max(voltage_x))
                    range_y = (np.min(voltage_y), np.max(voltage_y))
                    ranges = [range_x, range_y]

                    setpoints = self.raw_data[param_name].depends_on
                    meas.register_custom_parameter(
                        original_params[ip + 2].name,
                        label=original_params[ip + 2].label,
                        unit=original_params[1].unit,
                        paramtype="array",
                        setpoints=setpoints,
                    )
                    v_x_grid, v_y_grid = np.meshgrid(voltage_x, voltage_y)

                    result.append(
                        [
                            (setpoints[0], v_x_grid),
                            (setpoints[1], v_y_grid),
                            (param_name, signal.T),
                        ]
                    )

                with meas.run() as datasaver:
                    for r_i in range(len(self.readout_methods)):
                        datasaver.add_result(*result[r_i])

                    datasaver.dataset.add_metadata(
                        "snapshot", json.dumps(self.snapshot)
                    )
                    datasaver.dataset.add_metadata(
                        nt.meta_tag, json.dumps(self.nt_metadata)
                    )
                    datasaver.dataset.add_metadata(
                        "original_guid", json.dumps(self.guid)
                    )
                    logger.debug(
                        "New dataset created and populated.\n"
                        + "database: "
                        + str(segment_db_name)
                        + "ID: "
                        + str(datasaver.run_id)
                    )
                    segment_info[datasaver.run_id] = {}
                    segment_info[datasaver.run_id]["voltage_ranges"] = ranges

        return segment_info

    def get_triple_point_distances(self) -> Dict[str, List[Any]]:

        """
        Returns distances between peaks in heatmap. The idea behind this method
        was to extract triple point distances needed to fit the capacitance
        model. The latter is yet to be implemented and this current method is
        not robust.
        """
        relevant_distances = {}
        for read_meth in self.readout_methods:
            f_params = self.fit_parameters[read_meth]
            binary_neighborhood = f_params["binary_neighborhood"]
            noise_level = f_params["noise_level"]
            distance_threshold = f_params["distance_threshold"]

            data = self.filtered_data[read_meth]
            signal = data.values.T
            v_x = data[default_coord_names["voltage"][0]].values
            v_y = data[default_coord_names["voltage"][1]].values

            neighborhood = generate_binary_structure(2, binary_neighborhood)
            m_filter = maximum_filter(signal, footprint=neighborhood)
            detected_peaks = m_filter == signal

            background = signal <= noise_level
            eroded_background = binary_erosion(
                background, structure=neighborhood, border_value=1
            )

            detected_peaks[eroded_background == True] = False

            labeled_features = scm.label(detected_peaks)
            labels = labeled_features[0]
            n_features = labeled_features[1]
            if np.sum(n_features) > 1:
                coordinates = []

                for peak_id in range(1, n_features + 1):
                    indx = np.argwhere(labels == peak_id)
                    if len(indx) == 0:
                        logger.error("No peak found.")

                    else:
                        for ind in indx:
                            x_val = v_x[ind[1]]
                            y_val = v_y[ind[0]]
                            coordinates.append([x_val, y_val])
                coordinates_np: npt.NDArray[np.float64] = np.array(coordinates)

                # calculate distances between points, all to all
                all_combos = combinations(coordinates_np, 2)
                distances = [get_point_distances(*combo) for combo in all_combos]
                distances_arr = np.asarray(distances)

                relevant_indx = np.where(distances_arr[:, 0, 0] <= distance_threshold)[
                    0
                ]

                dist_list = [distances[indx] for indx in relevant_indx]
                relevant_distances[read_meth] = dist_list
            else:
                relevant_distances[read_meth] = []
        return relevant_distances

    def plot_fit(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        colorbar: Optional[matplotlib.colorbar.Colorbar] = None,
        save_figure: Optional[bool] = True,
        filename: Optional[str] = None,
        file_location: Optional[str] = None,
        plot_params: Optional[Dict[str, Any]] = None,
    ) -> AxesTuple:
        """"""
        if plot_params is None:
            plot_params = default_plot_params

        if self.triple_points is None:
            self.triple_points = self.get_triple_point_distances()

        fig_title = "Dotfit {}".format(self.guid)
        matplotlib.rcParams.update(plot_params)

        if ax is None:
            fig_size = copy.deepcopy(plot_params["figure.figsize"])
            fig_size[1] *= len(self.readout_methods) * 0.8  # type: ignore
            fig, ax = plt.subplots(
                len(self.readout_methods), 1, squeeze=False, figsize=fig_size
            )

        colorbars: List[matplotlib.colorbar.Colorbar] = []
        for r_i, read_meth in enumerate(self.readout_methods.keys()):
            voltage_x = self.data[read_meth]["voltage_x"].values
            voltage_y = self.data[read_meth]["voltage_y"].values
            signal = self.data[read_meth].values

            colormesh = ax[r_i, 0].pcolormesh(
                voltage_x,
                voltage_y,
                signal.T,
                shading="auto",
            )
            if colorbar is not None:
                colorbars.append(
                    ax[r_i, 0].figure.colorbar(
                        colormesh, ax=ax[r_i, 0], cax=colorbar.ax
                    )
                )
            else:
                divider = make_axes_locatable(ax[r_i, 0])
                cbar_ax = divider.append_axes("right", size="5%", pad=0.06)
                colorbars.append(fig.colorbar(colormesh, ax=ax[r_i, 0], cax=cbar_ax))

            colorbars[-1].set_label(self.get_plot_label(read_meth, 2), rotation=-270)

            ax[r_i, 0].set_xlabel(self.get_plot_label(read_meth, 0))
            ax[r_i, 0].set_ylabel(self.get_plot_label(read_meth, 1))
            ax[r_i, 0].set_title(fig_title)
            triple_points = np.asarray(self.triple_points[read_meth])
            try:
                ax[r_i, 0].scatter(
                    triple_points[:, 2, 0],
                    triple_points[:, 2, 1],
                    c="r",
                    label="electron triple point",
                )
                ax[r_i, 0].scatter(
                    triple_points[:, 3, 0],
                    triple_points[:, 3, 1],
                    c="b",
                    label="hole triple point",
                )
            except IndexError:
                pass

        ax[-1, 0].legend(loc="lower left", bbox_to_anchor=(0, 0))

        fig.tight_layout()

        if save_figure:
            if file_location is None:
                file_location = os.path.join(
                    nt.config["db_folder"], "dotfit", self.device_name
                )
            if not os.path.exists(file_location):
                os.makedirs(file_location)

            if filename is None:
                filename = "dotfit_" + str(self.guid)
            else:
                filename = os.path.splitext(filename)[0]

            path = os.path.join(file_location, filename + ".png")
            plt.savefig(path, format="png", dpi=600, bbox_inches="tight")

        return ax, colorbars


def get_point_distances(p1, p2):
    """return distance and points"""
    x1, y1 = p1
    x2, y2 = p2
    dist = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return (
        [dist, None],
        [abs(x2 - x1), abs(y2 - y1)],
        p1.tolist(),
        p2.tolist(),
    )
