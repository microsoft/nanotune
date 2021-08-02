import os
from typing import Callable, List, Optional, Tuple
import numpy.typing as npt
import numpy as np
import scipy.fftpack as fp
import scipy.signal as sg
from scipy.ndimage import generic_gradient_magnitude, sobel
from skimage.transform import AffineTransform, resize, rotate, warp

import nanotune as nt

transf_type = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def shear(
    original_image: npt.NDArray[np.float64],
    rotation: Optional[float] = None,
    shear: Optional[float] = None,
    translation: Optional[Tuple[float, float]] = None,
    scale: Optional[Tuple[float, float]] = None,
) -> npt.NDArray[np.float64]:

    if rotation is None:
        rotation = 0.4 * np.random.rand(1)[0]
    if shear is None:
        shear = 0.4 * np.random.rand(1)[0]

    original_shape = original_image.shape
    tform = AffineTransform(
        scale=scale,
        rotation=rotation,
        shear=shear,
        translation=translation,
    )

    return warp(original_image, tform, output_shape=(original_shape))


def random_crop(
    original_image: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    x_start = np.random.randint(np.floor(original_image.shape[0] / 2), size=1)
    x_range = np.random.randint(x_start + 20, original_image.shape[0], size=1)

    y_start = np.random.randint(np.floor(original_image.shape[1] / 2), size=1)
    y_range = np.random.randint(y_start + 20, original_image.shape[1], size=1)

    res = resize(original_image[x_start:x_range, y_start:y_range], (50, 50))
    return res


def random_rotation(
    original_image: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    angle = 360 * np.random.rand(1)
    return rotate(original_image, angle)


def random_flip(
    original_image: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if np.random.randint(0, 2, 1):
        # horizontal
        return original_image[:, ::-1]
    else:
        # vertical
        return original_image[::-1, :]


def no_transformation(
    original_image: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return original_image


def random_transformation(
    original_image: npt.NDArray[np.float64],
    transformations: Optional[List[transf_type]] = None,
    single: bool = True,
) -> npt.NDArray[np.float64]:
    """ """
    if transformations is None:
        transformations = [
            random_flip,
            random_rotation,
            random_crop,
            shear,
            no_transformation,
        ]
    if single:
        which_one = int(np.random.randint(0, len(transformations), 1))
        return transformations[which_one](original_image)
    else:
        trans_image = np.copy(original_image)
        for trans in range(len(transformations)):
            if np.random.randint(0, 2, 1):
                trans_image = transformations[trans](trans_image)
        return trans_image


def save_augmented_data(
    original_raw_data: npt.NDArray[np.float64],
    new_path: str,
    new_filename: str,
    mult_factor: int,
    write_period: int = 200,
    max_samples: int = 20000,
    data_types: List[str] = ["signal", "frequencies"],
) -> None:
    """ """
    # TODO: Is this method finished?
    total_counter = 0
    write_curr = 0
    shape = (50, 50)
    new_path = os.path.join(new_path, new_filename)

    index_sig = nt.config["core"]["data_types"]["signal"]
    index_freq = nt.config["core"]["data_types"]["frequencies"]
    index_grad = nt.config["core"]["data_types"]["gradient"]
    n_indx = len(nt.config["core"]["data_types"])

    condensed_data_all = np.empty((n_indx, 0, np.prod(shape) + 1))

    original_images = np.squeeze(original_raw_data[index_sig, :, :-1])
    print(original_images.shape)
    original_labels = original_raw_data[:, :, -1][0]
    print(original_labels.shape)

    if not os.path.exists(new_path):
        np.save(new_path, condensed_data_all)

    stop = False
    for it in range(mult_factor):
        for orig_image, orig_label in zip(original_images, original_labels):
            #         print(orig_image.shape)
            orig_image = orig_image.reshape(50, 50)
            condensed_data = np.empty((n_indx, 1, np.prod(shape) + 1))

            new_img = random_transformation(orig_image, single=False)
            condensed_data[index_sig, 0, :] = np.append(new_img.flatten(), orig_label)

            dtrnd = sg.detrend(new_img, axis=0)
            dtrnd = sg.detrend(dtrnd, axis=1)

            frequencies_res = fp.frequencies2(dtrnd)
            frequencies_res = np.abs(fp.frequenciesshift(frequencies_res))
            data_frq = resize(
                frequencies_res, (50, 50), anti_aliasing=True, mode="constant"
            ).flatten()

            condensed_data[index_freq, 0, :] = np.append(data_frq, orig_label)
            #             labels_all.append(orig_label)

            grad = generic_gradient_magnitude(new_img, sobel)
            gradient_resized = resize(
                grad, shape, anti_aliasing=True, mode="constant"
            ).flatten()
            condensed_data[index_grad, 0, :] = np.append(gradient_resized, orig_label)

            condensed_data_all = np.append(condensed_data_all, condensed_data, axis=1)

            write_curr += 1
            total_counter += 1
            if write_curr >= write_period:
                # save to file

                n = list(condensed_data_all.shape)
                n[-1] += 1

                previous_data = np.load(new_path)

                all_data = np.append(previous_data, condensed_data_all, axis=1)

                np.save(new_path, all_data)

                condensed_data_all = np.empty((n_indx, 0, np.prod(shape) + 1))
                write_curr = 0
            if total_counter >= max_samples:
                stop = True
                break
        if stop:
            break

    previous_data = np.load(new_path)

    all_data = np.append(previous_data, condensed_data_all, axis=1)

    np.save(new_path, all_data)

    # condensed_data_all = []
    # labels_all = []


# def generate_augmented_data(which: Optional[List]) -> None:
#     """
#     """
#     cm_raw = np.load('/Users/jana/Documents/code/nanotune/measurements/databases/noiseless_data.npy')
#     folder = nt.config['db_folder']
#     # fnames = ['augmented_cm_data1.npy', 'augmented_cm_data2.npy', 'augmented_cm_data3.npy']
#     fnames = ['test.npy']
#     for fname in fnames:
#         save_augmented_data(cm_raw,
#                             folder,
#                             fname,
#                             7,
#                             write_period=200,
#     #                         data_types=['signal', 'frequencies'],
#                         )

#     qf_raw = np.load('/Users/jana/Documents/code/nanotune/measurements/databases/qflow_data_large.npy')
#     folder = nt.config['db_folder']
#     fnames = ['augmented_qf_data1.npy', 'augmented_qf_data2.npy', 'augmented_qf_data3.npy']

#     for fname in fnames:
#         save_augmented_data(qf_raw,
#                             folder,
#                             fname,
#                             2,
#                             write_period=200,
#     #                         data_types=['signal', 'frequencies'],
#                         )

#     folder = nt.config['db_folder']
#     exp_data_fnames = ['clean_exp_dots.npy', 'exp_dots_corrected.npy', 'exp_dots_minus_clean.npy']
#     fnames = [['augmented_clean_exp_dots1.npy', 'augmented_clean_exp_dots2.npy', 'augmented_clean_exp_dots2.npy'],
#             ['augmented_exp_dots_corrected1.npy', 'augmented_exp_dots_corrected2.npy', 'augmented_exp_dots_corrected3.npy'],
#             ['augmented_exp_dots_minus_clean1.npy', 'augmented_exp_dots_minus_clean2.npy', 'augmented_exp_dots_minus_clean3.npy']]

#     for original_file, new_files in zip(exp_data_fnames, fnames):
#         exp_raw = np.load(os.path.join(folder, original_file))
#         for fname in new_files:
#             save_augmented_data(exp_raw,
#                                 folder,
#                                 fname,
#                                 30,
#                                 write_period=200,
#         #                         data_types=['signal', 'frequencies'],
#                             )
