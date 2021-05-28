import logging
import os
from typing import Dict, List, Optional

import numpy as np
import scipy.fft as fp
from qcodes.dataset.experiment_container import load_by_id
from qcodes.dataset.measurements import Measurement
from scipy.ndimage import generic_gradient_magnitude, sobel
from skimage.transform import resize

import nanotune as nt
from nanotune.data.databases import get_dataIDs
from nanotune.data.dataset import Dataset
from nanotune.fit.dotfit import DotFit

logger = logging.getLogger(__name__)
N_2D = nt.config["core"]["standard_shapes"]["2"]
NT_LABELS = list(dict(nt.config["core"]["labels"]).keys())


def export_label(
    ds_label: List["str"],
    quality: int,
    category: str,
) -> int:
    """
    # Condense dot labels to:
    # 0: poor singledot
    # 1: good singledot
    # 2: poor doubledot
    # 3: good doubledot

    All others remain the same i.e. this method oly returns the quality
    """
    good = bool(quality)
    if category == "dotregime":
        singledot = True if "singledot" in ds_label else False
        doubledot = True if "doubledot" in ds_label else False

        if not good and singledot:
            new_label = 0
        if good and singledot:
            new_label = 1
        if not good and doubledot:
            new_label = 2
        if good and doubledot:
            new_label = 3

    elif category in ["outerbarriers", "pinchoff", "singledot", "doubledot"]:
        if len(ds_label) == 1 and ds_label[0] == category:
            new_label = good
        else:
            print(category)
            print(ds_label)
            logger.warning("Wrong label-category combination in export_label.")
            raise ValueError
    else:
        logger.error(
            "Do not know how to export/condense labels. Please "
            + "update export_label in export_data.py."
        )
        raise ValueError

    return new_label


def prep_data(
    dataset: nt.Dataset,
    category: str,
) -> np.array:
    """
    Remove nans, normalize by normalization_constants and reshape into
    target shape
    shape convention:
    shape =  datatypes, #samples, #features]
    We return 1 sample and 2 datatypes
    """
    assert category in nt.config["core"]["features"].keys()
    if len(dataset.power_spectrum) == 0:
        dataset.compute_power_spectrum()

    condensed_data_all = []

    for readout_method in dataset.readout_methods.keys():
        signal = dataset.data[readout_method].values
        dimension = dataset.dimensions[readout_method]

        shape = tuple(nt.config["core"]["standard_shapes"][str(dimension)])
        condensed_data = np.empty(
            (len(nt.config["core"]["data_types"]), 1, np.prod(shape))
        )

        relevant_features = nt.config["core"]["features"][category]
        features = []

        if dataset.features:
            if all(isinstance(i, dict) for i in dataset.features.values()):
                for feat in relevant_features:
                    features.append(dataset.features[readout_method][feat])
            else:
                for feat in relevant_features:
                    features.append(dataset.features[feat])

        # double check if current range is correct:
        if np.max(signal) > 1:
            min_curr = np.min(signal)
            max_curr = np.max(signal)
            signal = (signal - min_curr) / (max_curr - min_curr)
            # assume we are talking dots and high current was not actually
            # device_max_signal
            dataset.data[readout_method].values = signal * 0.3
            dataset.compute_power_spectrum()

        data_resized = resize(
            signal, shape, anti_aliasing=True, mode="edge"
        ).flatten()

        grad = generic_gradient_magnitude(signal, sobel)
        gradient_resized = resize(
            grad, shape, anti_aliasing=True, mode="constant"
        ).flatten()
        power = dataset.power_spectrum[readout_method].values
        frequencies_resized = resize(
            power, shape, anti_aliasing=True, mode="constant"
        ).flatten()

        pad_width = len(data_resized.flatten()) - len(features)
        features = np.pad(
            features,
            (0, pad_width),
            "constant",
            constant_values=nt.config["core"]["fill_value"],
        )

        index = nt.config["core"]["data_types"]["signal"]
        condensed_data[index, 0, :] = data_resized

        index = nt.config["core"]["data_types"]["frequencies"]
        condensed_data[index, 0, :] = frequencies_resized

        index = nt.config["core"]["data_types"]["gradient"]
        condensed_data[index, 0, :] = gradient_resized

        index = nt.config["core"]["data_types"]["features"]
        condensed_data[index, 0, :] = features

        condensed_data_all.append(condensed_data)

    return condensed_data_all


def export_data(
    category: str,
    db_names: List[str],
    stages: List[str],
    skip_ids: Optional[Dict[str, List[int]]] = None,
    quality: Optional[int] = None,
    filename: Optional[str] = None,
    db_folder: Optional[str] = None,
) -> None:
    """"""
    assert isinstance(db_names, list)
    assert isinstance(stages, list)

    if db_folder is None:
        db_folder = nt.config["db_folder"]

    if category in ["pinchoff", "coulomboscillation"]:
        dim = 1
    elif category in [
        "dotregime",
        "singledot",
        "doubledot",
        "coulombdiamonds",
    ]:
        dim = 2
    else:
        logger.error(
            "Trying to export data of"
            + " a category: {}/".format(category)
            + " Please update utils/export_data.py and tell me"
            + " the dimensions of the data "
        )

    shape = tuple(nt.config["core"]["standard_shapes"][str(dim)])
    condensed_data_all = np.empty(
        (len(nt.config["core"]["data_types"]), 0, np.prod(shape))
    )

    relevant_ids: Dict[str, List[int]] = {}
    for db_name in db_names:
        relevant_ids[db_name] = []
        nt.set_database(db_name, db_folder)
        for stage in stages:
            try:
                if quality is None:
                    relevant_ids[db_name] += nt.get_dataIDs(
                        db_name, stage, db_folder=db_folder
                    )
                else:
                    relevant_ids[db_name] += nt.get_dataIDs(
                        db_name, stage, quality=quality, db_folder=db_folder
                    )
            except Exception as e:
                logger.error(
                    """Unable to load relevant ids
                in {}""".format(
                        db_name
                    )
                )
                logger.error(e)
                break

    labels_exp = []

    for db_name, dataids in relevant_ids.items():
        nt.set_database(db_name, db_folder)
        skip_us = []
        if skip_ids is not None:
            try:
                skip_us = skip_ids[db_name]
            except KeyError:
                logger.warning("No data IDs to skip in {}.".format(db_name))

        for d_id in dataids:
            if d_id not in skip_us:
                df = Dataset(d_id, db_name, db_folder=db_folder)

                condensed_data = prep_data(df, category)
                condensed_data_all = np.append(
                    condensed_data_all, condensed_data[0], axis=1
                )
                new_label = export_label(df.label, df.quality, category)
                labels_exp.append(new_label)

    n = list(condensed_data_all.shape)
    n[-1] += 1

    data_w_labels = np.zeros(n)
    data_w_labels[:, :, -1] = labels_exp
    data_w_labels[:, :, :-1] = condensed_data_all

    if filename is None:
        filename = "_".join(stages)
    path = os.path.join(db_folder, filename)
    np.save(path, data_w_labels)


def correct_normalizations(
    filename: str,
    db_folder: Optional[str] = None,
) -> None:
    """"""
    if db_folder is None:
        db_folder = nt.config["db_folder"]

    path = os.path.join(db_folder, filename)

    all_data = np.load(path)

    data = all_data[:, :, :-1]
    labels = all_data[:, :, -1]

    sg_indx = nt.config["core"]["data_types"]["signal"]

    images = np.copy(data[sg_indx])
    images = images.reshape(images.shape[0], -1)

    high_current_images = np.max(images, axis=1)
    high_current_ids = np.where(high_current_images > 1)[0]

    # print(len(high_current_ids))

    for exid in high_current_ids:
        # print(np.max(data[sg_indx, exid]))
        sig = data[sg_indx, exid]
        sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig))
        sig = sig * 0.3  # assume it's dots and highest current is not max current
        data[sg_indx, exid] = sig

        freq_spect = fp.fft2(sig.reshape(50, 50))
        freq_spect = np.abs(fp.fftshift(freq_spect))

        grad = generic_gradient_magnitude(sig.reshape(50, 50), sobel)

        index = nt.config["core"]["data_types"]["frequencies"]
        data[index, exid, :] = freq_spect.flatten()

        index = nt.config["core"]["data_types"]["gradient"]
        data[index, exid, :] = grad.flatten()

    n = list(data.shape)
    n[-1] += 1

    data_w_labels = np.zeros(n)
    data_w_labels[:, :, -1] = labels
    data_w_labels[:, :, :-1] = data

    path = os.path.join(db_folder, filename)
    np.save(path, data_w_labels)


# def subsample_2Ddata(
#     db_names: List[str],
#     target_db_name: str,
#     stages: List[str],
#     qualities: List[int],
#     original_db_folders: Optional[List[str]] = None,
#     targe_db_folder: Optional[str] = None,
#     n_slices: int = 5,
#     temp_sub_size: int = 20,
# ) -> None:
#     """
#     TODO: Fix subsample_2Ddata: Implement a method to increase the number of
#     datasets one can use for classifier training. "Cut out" random sub_images
#     from exisitng data. These don't need to be saved in a db.
#     """
#     if original_db_folder is None:
#         original_db_folder = [nt.config["db_folder"]]*len(db_names)
#     if targe_db_folder is None:
#         targe_db_folder = nt.config["db_folder"]

#     for original_db, original_db_folder in zip(db_names, original_db_folder):
#         nt.set_database(original_db, original_db_folder)

#         relevant_ids: List[int] = []
#         for stage in stages:
#             for quality in qualities:
#                 temp_ids = nt.get_dataIDs(
#                     original_db, stage,
#                     quality=quality,
#                     db_folder=original_db_folder,
#                 )
#                 temp_ids = list(filter(None, temp_ids))
#                 relevant_ids += temp_ids

#         for did in relevant_ids:
#             ds = nt.Dataset(did, original_db, db_folder=original_db_folder)
#             current_label = dict.fromkeys(NT_LABELS, 0)
#             for lbl in ds.label:
#                 current_label[lbl] = 1

#             with nt.switch_database(target_db_name, target_db_folder):
#                 meas = Measurement()
#                 meas.register_custom_parameter(
#                     ds.qc_parameters[0].name,
#                     label=ds.qc_parameters[0].label,
#                     unit=ds.qc_parameters[0].unit,
#                     paramtype="array",
#                 )

#                 meas.register_custom_parameter(
#                     ds.qc_parameters[1].name,
#                     label=ds.qc_parameters[1].label,
#                     unit=ds.qc_parameters[1].unit,
#                     paramtype="array",
#                 )

#                 for ip, param_name in enumerate(list(ds.raw_data.data_vars)):
#                     coord_names = list(ds.raw_data.coords)
#                     x_crd_name = coord_names[0]
#                     y_crd_name = coord_names[1]

#                     x = ds.raw_data[param_name][x_crd_name].values
#                     y = ds.raw_data[param_name][y_crd_name].values
#                     signal = ds.raw_data[param_name].values


#                     for _ in range(n_slices):
#                         x0 = random.sample(range(0, N_2D[0] - temp_sub_size), 1)[0]
#                         y0 = random.sample(range(0, N_2D[1] - temp_sub_size), 1)[0]

#                         sub_x = np.linspace(x[x0], x[x0 + temp_sub_size], N_2D[0])
#                         sub_y = np.linspace(y[y0], y[y0 + temp_sub_size], N_2D[1])

#                         sub_x, sub_y = np.meshgrid(sub_x, sub_y)


#                         output = []
#                         for mid, meas_param in enumerate(ds.qc_parameters[2:]):
#                             s1 = ds.qc_parameters[0].name
#                             s2 = ds.qc_parameters[1].name
#                             meas.register_custom_parameter(
#                                 meas_param.name,
#                                 label=meas_param.label,
#                                 unit=meas_param.unit,
#                                 paramtype="array",
#                                 setpoints=[s1, s2],
#                             )

#                             sub_data = signal[mid][
#                                 x0 : x0 + temp_sub_size, y0 : y0 + temp_sub_size
#                             ]
#                             sub_data = resize(sub_data, N_2D)
#                             output.append([meas_param.name, sub_data])

#                         with meas.run() as datasaver:
#                             datasaver.add_result(
#                                 (ds.qc_parameters[0].name, sub_x),
#                                 (ds.qc_parameters[1].name, sub_y),
#                                 *output,  # type: ignore
#                             )

#                         new_ds = load_by_id(datasaver.run_id)
#                         new_ds.add_metadata("snapshot", json.dumps(ds.snapshot))
#                         new_ds.add_metadata("original_guid", ds.guid)

#                         for key, value in current_label.items():
#                             new_ds.add_metadata(key, json.dumps(value))
