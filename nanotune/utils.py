import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, no_type_check

import matplotlib.pyplot as plt
import numpy as np
from qcodes.dataset.experiment_container import load_by_id

import nanotune as nt

logger = logging.getLogger(__name__)


def get_path(filename: str) -> str:
    """
    Get path to file, which is expected to be in
    nt.config['db_folder']
    """
    return os.path.join(nt.config["db_folder"], filename)


@no_type_check
def flatten_list(nested_list: List[Any]) -> Any:
    """
    Returns an operator/iterator iterating through a flattened list
    """
    for i in nested_list:
        if isinstance(i, (list, tuple)):
            for j in flatten_list(i):
                yield j
        else:
            yield i


def get_recursively(
    search_dict: Dict[str, Any],
    parameter_name: str,
    value_field: str = "value",
) -> List[str]:
    """
    Recursively searches a QCoDeS metadata dict for the value of a given
    parameter.
    """
    param_values = []

    for key, value in search_dict.items():
        if value == parameter_name:
            param_values.append(search_dict[value_field])

        elif isinstance(value, dict):
            results = get_recursively(value, parameter_name)
            for result in results:
                param_values.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, parameter_name)
                    for another_result in more_results:
                        param_values.append(another_result)

    return param_values


def get_param_values(
    qc_run_id: int,
    db_name: str,
    db_folder: Optional[str] = None,
    return_meta_add_on: Optional[bool] = False,
) -> Tuple[List[List[str]], List[List[str]]]:
    """"""

    if db_folder is None:
        db_folder = nt.config["db_folder"]

    nt.set_database(db_name)
    ds = load_by_id(qc_run_id)

    nt_metadata = json.loads(ds.get_metadata(nt.meta_tag))
    snapshot = json.loads(ds.get_metadata("snapshot"))["station"]

    device_name = nt_metadata["device_name"]
    device_snap = snapshot["instruments"][device_name]
    submods = device_snap["submodules"].keys()
    param_values = [["Parameter", "Value"]]
    for submod in submods:
        gate_val = device_snap["submodules"][submod]["parameters"]
        try:
            gate_val = gate_val["dc_voltage"]["value"]
        except KeyError:
            gate_val = gate_val["state"]["value"]
        param_values.append([submod, gate_val])

    features = []

    if return_meta_add_on:
        features = [["Feature", "Value"]]

        param_values.append(["db_name", db_name])
        param_values.append(["guid", ds.guid])
        for name, v in nt_metadata.items():
            if name == "elapsed_time" and v is not None:
                m, s = divmod(v, 60)
                h, m = divmod(m, 60)
                v = "{}h {}min {}s".format(h, m, s)
                param_values.append([name, v])
            elif name == "features":
                for fname, fval in v.items():
                    features.append([fname, fval])
            elif name == "normalization_constants":
                param_values.append(["dc normalization", str(v["transport"])])
                param_values.append(["rf normalization", str(v["rf"])])
            else:
                if type(v) == list:
                    param_values.append([name, *v])
                else:
                    param_values.append([name, v])

    return param_values, features


def format_axes(axes: List[plt.Axes], color: str, linewidth: float):
    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(linewidth)

        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_tick_params(direction="out", color=color)

    return axes
