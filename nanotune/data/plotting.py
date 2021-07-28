import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import nanotune as nt
from nanotune.data.dataset import Dataset, default_coord_names

AxesTuple = Tuple[matplotlib.axes.Axes, List[matplotlib.colorbar.Colorbar]]
plot_params_type = Dict[str, Union[str, float, int, bool, List[float]]]

default_plot_params: plot_params_type = {
    "backend": "ps",
    # 'text.latex.preamble': [r'\usepackage{gensymb}'],
    "image.origin": "lower",
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.labelweight": 10,
    "axes.edgecolor": "grey",
    "axes.labelpad": 0.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.numpoints": 1,
    "legend.markerscale": 1,
    #                'legend.fontsize': 'x-small',
    #               'text.fontsize': 8,
    "font.size": 10,
    "lines.linewidth": 0.6,
    "lines.markersize": 5,
    "savefig.dpi": 300,
    "axes.grid": False,
    "image.interpolation": "nearest",
    "text.usetex": False,
    "legend.fontsize": 10,
    "legend.labelspacing": 0.5,
    "legend.framealpha": 0.8,
    "figure.figsize": [7.0, 5.0],
    "font.family": "serif",
    # 'pcolor.shading': 'auto,
}

lightblue = "#6699CC"  # (255, 153, 204)
blue = "#336699"  # (51, 102, 153)
darkblue = "#264D73"  # (38, 77, 115)

cyan = "#33BBEE"  # (51, 187, 238)

lightteal = "#00E6E6"  # (0, 230, 230)
teal = "#009988"  # (0, 153, 136)
darkteal = "#006666"  # (0, 102, 102)

orange = "#EE7733"  # (238, 119, 51)

lightred = "#FF531A"  # (255, 83, 26)
red = "#CC3311"  # (204, 51, 17)
darkred = "#802000"  # (128, 32, 0)

magenta = "#EE3377"  # (238, 51, 119)
grey = "#BBBBBB"  # (187, 187, 187)


custom_cm = LinearSegmentedColormap.from_list(
    "vivid_colorblind", [darkblue, cyan, teal, red, orange]
)
plt.register_cmap(cmap=custom_cm)
matplotlib.rcParams["image.cmap"] = "vivid_colorblind"

colors = [
    blue,
    red,
    cyan,
    orange,
    teal,
    lightblue,
    lightteal,
    lightred,
    darkblue,
    darkteal,
    darkred,
    grey,
]
matplotlib.rcParams["axes.prop_cycle"] = cycler(color=colors)
# ['#003399', '#FF6633', '#996699', '#99CCFF', '#EE442F', '#F4D4D4',
# '#63ACBE', '#9C9EB5', '#FDF0F2', '#ABC3C9'])

colors_dict = {
    "lightblue": lightblue,
    "blue": blue,
    "darkblue": darkblue,
    "cyan": cyan,
    "lightteal": lightteal,
    "teal": teal,
    "darkteal": darkteal,
    "orange": orange,
    "lightred": lightred,
    "red": red,
    "darkred": darkred,
    "magenta": magenta,
    "grey": grey,
}


def plot_dataset(
    qc_run_id: int,
    db_name: str,
    save_figures: bool = True,
    db_folder: Optional[str] = None,
    plot_filtered_data: bool = False,
    plot_params: Optional[plot_params_type] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    colorbar: Optional[matplotlib.colorbar.Colorbar] = None,
    filename: Optional[str] = None,
    file_location: Optional[str] = None,
) -> AxesTuple:
    """
    If to be saved and no file location specified, the figure will be saved at
    os.path.join(nt.config['db_folder'], 'tuning_results', dataset.device_name)
    in both eps and png
    """

    if plot_params is None:
        plot_params = default_plot_params
    matplotlib.rcParams.update(plot_params)
    if db_folder is None:
        _, db_folder = nt.get_database()

    dataset = Dataset(qc_run_id, db_name, db_folder=db_folder)

    if plot_filtered_data:
        data = dataset.filtered_data
    else:
        data = dataset.data

    if ax is None:
        fig_size = copy.deepcopy(plot_params["figure.figsize"])
        fig_size[1] *= len(dataset.data) * 0.8  # type: ignore
        fig, ax = plt.subplots(
            len(dataset.data),
            1,
            squeeze=False,
            figsize=fig_size,
        )

        colorbars: List[matplotlib.colorbar.Colorbar] = []

    fig_title = dataset.guid

    for r_i, read_meth in enumerate(dataset.readout_methods):
        c_name = default_coord_names["voltage"][0]
        voltage_x = data[read_meth][c_name].values
        signal = data[read_meth].values.T

        if dataset.dimensions[read_meth] == 1:
            colorbar = None
            ax[r_i, 0].plot(
                voltage_x,
                signal,
                zorder=6,
            )
            ax[r_i, 0].set_xlabel(dataset.get_plot_label(read_meth, 0))
            ax[r_i, 0].set_ylabel(dataset.get_plot_label(read_meth, 1))
            ax[r_i, 0].set_title(str(fig_title))

            divider = make_axes_locatable(ax[r_i, 0])
            cbar_ax = divider.append_axes("right", size="5%", pad=0.06)
            cbar_ax.set_facecolor("none")
            for caxis in ["top", "bottom", "left", "right"]:
                cbar_ax.spines[caxis].set_linewidth(0)
            cbar_ax.set_xticks([])
            cbar_ax.set_yticks([])
            colorbars.append(colorbars)

            ax[r_i, 0].figure.tight_layout()

        elif dataset.dimensions[read_meth] == 2:
            c_name = default_coord_names["voltage"][1]
            voltage_y = data[read_meth][c_name].values
            colormesh = ax[r_i, 0].pcolormesh(
                voltage_x,
                voltage_y,
                signal,
                shading="auto",
            )

            if colorbar is not None:
                colorbars.append(
                    ax[r_i, 0].figure.colorbar(
                        colormesh, ax=ax[r_i, 0], cax=colorbar.ax
                    )
                )
            else:
                # colorbar = fig.colorbar(colormesh, ax=ax[r_i, 0])
                divider = make_axes_locatable(ax[r_i, 0])
                cbar_ax = divider.append_axes("right", size="5%", pad=0.06)
                colorbars.append(
                    fig.colorbar(
                        colormesh,
                        ax=ax[r_i, 0],
                        cax=cbar_ax,
                    )
                )
            colorbars[-1].set_label(
                dataset.get_plot_label(read_meth, 2),
                rotation=-270,
            )

            ax[r_i, 0].set_xlabel(dataset.get_plot_label(read_meth, 0))
            ax[r_i, 0].set_ylabel(dataset.get_plot_label(read_meth, 1))
            ax[r_i, 0].set_title(str(fig_title))

            ax[r_i, 0].figure.tight_layout()

        else:
            raise NotImplementedError

    if save_figures:
        if file_location is None:
            file_location = os.path.join(
                nt.config["db_folder"], "tuning_results", dataset.device_name
            )
        if not os.path.exists(file_location):
            os.makedirs(file_location)

        if filename is None:
            filename = "dataset_" + str(dataset.guid)
        else:
            filename = os.path.splitext(filename)[0]

        path = os.path.join(file_location, filename + ".png")
        plt.savefig(path, format="png", dpi=600, bbox_inches="tight")
    return ax, colorbars
