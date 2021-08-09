import logging
import os
from typing import Dict, List, Optional

import numpy as np
import scipy.fft as fp
from scipy.ndimage import generic_gradient_magnitude, sobel
from skimage.transform import resize

import nanotune as nt

logger = logging.getLogger(__name__)
N_2D = nt.config["core"]["standard_shapes"]["2"]
NT_LABELS = list(dict(nt.config["core"]["labels"]).keys())


def export_label(
    ds_label: List["str"],
    quality: int,
    category: str,
) -> int:
    """Merges binary labels of single and double dot qualities to a single
    label. Only if `dotregime` is specified, for all other the initial
    quality labels are returned.
    It translates a dot regime label to: 0 - poor singledot, 1 - good singledot,
    2 - poor doubledot, 3 - good doubledot.

    Returns:
        int: new label.
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
    flip_data: bool = False
) -> List[List[List[float]]]:
    """Prepares data for classification.
    It combines normalized data, its gradient, Fourier frequencies and
    extracted features into a multidimensional list.
    All sublists are reshaped to a standard shape, defined in config.json
    under the `standard_shapes` key.

    Args:
        dataset: instance of nanotune dataset whose data should be prepared.
        category: as which category/type of data, e.g. `pinchoff`, `singledot`
            etc, it should be treated.
        flip_data: whether data should be flipped. Used to simulate pinchoff
            curves measured in rf sensing.

    Returns:
        list: multidimensional list. First sublist is normalized data,
            second Fourier frequencies, third gradient and fourth features.
    """
    assert category in nt.config["core"]["features"].keys()
    if len(dataset.power_spectrum) == 0:
        dataset.compute_power_spectrum()

    condensed_data_all = []

    for readout_method in dataset.readout_methods.keys():
        signal = dataset.data[readout_method].values
        if flip_data:
            signal = np.flip(signal)
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
        condensed_data[index, 0, :] = data_resized.tolist()

        index = nt.config["core"]["data_types"]["frequencies"]
        condensed_data[index, 0, :] = frequencies_resized.tolist()

        index = nt.config["core"]["data_types"]["gradient"]
        condensed_data[index, 0, :] = gradient_resized.tolist()

        index = nt.config["core"]["data_types"]["features"]
        condensed_data[index, 0, :] = features

        condensed_data_all.append(condensed_data.tolist())

    return condensed_data_all


def export_data(
    category: str,
    db_names: List[str],
    stages: List[str],
    skip_ids: Optional[Dict[str, List[int]]] = None,
    add_flipped_data: bool = False,
    quality: Optional[int] = None,
    filename: Optional[str] = None,
    db_folder: Optional[str] = None,
) -> None:
    """Exports condensed data to a numpy file.

    Args:
        category:
        db_names:
        stages:
    """
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
                try:
                    df = nt.Dataset(d_id, db_name, db_folder=db_folder)
                    condensed_data = prep_data(df, category)
                    new_label = export_label(df.label, df.quality, category)
                    condensed_data_all = np.append(
                        condensed_data_all, condensed_data[0], axis=1
                    )
                    labels_exp.append(new_label)

                    if add_flipped_data:
                        condensed_data = prep_data(
                            df, category, flip_data=True
                        )
                        new_label = export_label(
                            df.label, df.quality, category
                        )
                        condensed_data_all = np.append(
                            condensed_data_all, condensed_data[0], axis=1
                        )
                        labels_exp.append(new_label)
                except (IndexError, ValueError, TypeError) as i_err:
                    print(db_name)
                    print(d_id)
                    print(i_err)

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
