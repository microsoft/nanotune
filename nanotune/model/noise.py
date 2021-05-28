import os
import logging

from typing import List, Optional, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

import scipy.fftpack as fp
from skimage.transform import resize

import nanotune as nt

logger = logging.getLogger(__name__)

NOISE_TYPES = ["white", "rnt", "one_over_f", "random_blobs", "current_drop"]

DEFAULT_FILES = {
    "white": "white_noise.npy",
    "rnt": "random_telegraph_noise.npy",
    "one_over_f": "one_over_f_noise.npy",
    "random_blobs": "random_blobs.npy",
    "current_drop": "current_drop.npy",
}

N_2D = nt.config["core"]["standard_shapes"]["2"]


def load_noise(
    noise_types: List[str],
    number_of_samples: int,
    files: Optional[Dict[str, str]] = None,
    folder: Optional[str] = None,
) -> np.ndarray:
    """
    Note: complex numbers are cast into floats here, might need to fix this
    of frequencies do not give desired result
    """
    if files is None:
        files = DEFAULT_FILES

    if folder is None:
        folder = nt.config["db_folder"]

    all_noise = {}
    # np.zeros((len(noise_types), 2, number_of_samples, *N_2D))
    # noise_idx = []
    for ntype in noise_types:
        if ntype not in NOISE_TYPES:
            logger.error(
                "Unknown noise type. Choose one of the following: "
                + " {}".format(", ".join(NOISE_TYPES))
            )
            raise ValueError
        NOISE_TYPES.index(ntype)

        raw_noise = np.load(os.path.join(folder, DEFAULT_FILES[ntype]))
        raw_noise = np.reshape(raw_noise[0, :, :], (raw_noise.shape[1], *N_2D))

        raw_noise = raw_noise[
            np.random.choice(len(raw_noise), number_of_samples, replace=True).astype(
                int
            )
        ]
        raw_noise_freq = fp.frequencies2(raw_noise)
        raw_noise_freq = fp.frequenciesshift(raw_noise_freq)

        all_noise[ntype] = np.zeros((2, number_of_samples, *N_2D))
        all_noise[ntype][0] = raw_noise
        all_noise[ntype][1] = raw_noise_freq.real

    return all_noise


def add_noise(
    original_data: np.ndarray,
    noise_types: List[str],
    max_strength: List[float],
    n_samples: Optional[int] = None,
    in_current: bool = True,
    min_strength: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    assert len(noise_types) == len(max_strength)

    noisy_data = np.copy(original_data)
    m = noisy_data.shape[0]

    if min_strength is None:
        min_strength = [0] * len(noise_types)

    if n_samples is None:
        n_samples = m

    if not in_current:
        org_max = np.max(noisy_data.reshape(m, -1), axis=1)
        noisy_data = np.reshape(noisy_data, (m, *N_2D))
        noisy_freq = fp.frequencies2(noisy_data)
        noisy_freq = fp.frequenciesshift(noisy_freq)

    noisy_data = np.reshape(noisy_data, (m, -1))
    raw_noise = load_noise(noise_types, m)

    for inn, ntype in enumerate(noise_types):
        if ntype not in NOISE_TYPES:
            logger.error(
                "Unknown noise type. Choose one of the following: "
                + " {}".format(", ".join(NOISE_TYPES))
            )
            raise ValueError

        # if ntype in ['current_drop', 'random_blobs']:
        #     min_strength[inn] = np.min(1,  max_strength[inn])

        amp = np.random.uniform(min_strength[inn], max_strength[inn], (n_samples, 1))
        amp = np.append(amp, np.zeros((m - n_samples, 1)))
        p = np.random.permutation(len(amp))
        amp = amp[p].reshape(m, 1)

        if in_current:
            noise = raw_noise[ntype][0]
            old_max = np.max(noisy_data, axis=1).reshape(m, 1)
            if ntype in ["current_drop", "random_blobs"]:
                # amp[amp=0] = 1
                noise = amp * noise.reshape(m, -1)
                idx = np.where(amp == 0)[0]
                noise[idx] = np.ones(noise.shape[-1])
                noisy_data = noisy_data * noise
            else:
                noisy_data = noisy_data + amp * noise.reshape(m, -1)

            new_max = np.max(noisy_data, axis=1).reshape(m, 1)
            noisy_data = noisy_data * old_max / new_max
        else:
            noise = raw_noise[ntype][1]
            if ntype in ["current_drop", "random_blobs"]:
                noise = amp * noise.reshape(m, -1)
                idx = np.where(amp == 0)[0]
                noise[idx] = np.ones(noise.shape[-1])
                noisy_freq = noisy_freq * noise
            else:
                noisy_freq = noisy_freq + amp * noise.reshape(m, -1)

    if in_current:
        noisy_data = np.reshape(noisy_data, (m, *N_2D))
        noisy_freq = fp.frequencies2(noisy_data)
        noisy_freq = fp.frequenciesshift(noisy_freq)

    else:
        noisy_freq = np.reshape(noisy_freq, (m, *N_2D))
        noisy_data = np.abs(fp.ifrequencies2(noisy_freq))

        new_max = np.max(noisy_data, axis=1).reshape(m, 1)
        noisy_data = noisy_data * org_max / new_max

    noisy_freq = np.reshape(noisy_freq, (m, *N_2D, 1))
    noisy_data = np.reshape(noisy_data, (m, *N_2D, 1))

    # noisy_data = (noisy_data - np.min(noisy_data))/(np.max(noisy_data) - np.min(noisy_data)) * 0.3
    # nmin = np.min(images, axis=(1, 2)).reshape(-1, 1)
    # nmax = np.max(images, axis=(1, 2)).reshape(-1, 1)
    # images = images.reshape(images.shape[0], -1)
    # images = (images - nmin)/(nmax - nmin)
    return noisy_data, noisy_freq


def add_random_charge_shifts(
    original_data: np.ndarray,
    number_of_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    data = np.copy(original_data)
    data_idx = np.random.choice(
        original_data.shape[0], number_of_samples, replace=False
    ).astype(int)
    org_shape = data.shape

    for idx in data_idx:
        ex_data = np.squeeze(data[idx])

        n_diff = np.random.randint(5, 9, 1)
        min_d = int(np.floor(n_diff / 2))
        n_step = np.random.randint(2, ex_data.shape[0], 1)[0]

        transpose = np.random.randint(2, size=1)[0]

        if transpose:
            new_img1 = np.concatenate(
                (
                    ex_data[:n_step, :],
                    ex_data[n_step - min_d : n_step + min_d, :],
                    ex_data[n_step + min_d :, :],
                ),
                axis=0,
            )
        else:
            new_img1 = np.concatenate(
                (
                    ex_data[:, :n_step],
                    ex_data[:, n_step - min_d : n_step + min_d],
                    ex_data[:, n_step + min_d :],
                ),
                axis=1,
            )
        new_img1 = resize(new_img1, (50, 50))

        data[idx] = new_img1.reshape(1, *org_shape[1:])

    m = data.shape[0]

    data = np.reshape(data, (m, *N_2D))
    freq_data = fp.frequencies2(data)
    freq_data = np.abs(fp.frequenciesshift(freq_data))

    data = data.reshape(*org_shape)
    freq_data = freq_data.reshape(*org_shape)

    return data, freq_data


# def add_noise_in_freq_domain(files: List) -> None:
#     """
#     """
#     # # syn_data_freq = fp.frequencies2(data_current)
#     # # syn_data_freq = fp.frequenciesshift(syn_data_freq)

#     # noisy_freqs = np.copy(syn_freq)

#     # m = noisy_freqs.shape[0]

#     # amp = np.random.uniform(0, 0.3, (3, m))
#     # rnt_amp = np.random.uniform(0, 0.03, (m))
#     # noisy_freqs = np.reshape(noisy_freqs, (m, -1))

#     # # noisy_freqs = noisy_freqs + amp[0].reshape(-1, 1)*white_freq_selection.reshape(m, -1)
#     # noisy_freqs = noisy_freqs + amp[1].reshape(-1, 1)*one_over_f_freq_selection.reshape(m, -1)
#     # # noisy_freqs = noisy_freqs + amp[2].reshape(-1, 1)*random_blobs_freq_selection.reshape(m, -1)*0.1
#     # # noisy_freqs = noisy_freqs + rnt_amp.reshape(-1, 1)*rnt_freq_selection.reshape(m, -1)
#     # # noisy_freqs = noisy_freqs*current_drop_freq_selection.reshape(m, -1)

#     # noisy_freqs = np.reshape(noisy_freqs, (m, 50, 50))

#     # images = np.abs(fp.ifrequencies2(noisy_freqs))

#     # images = images*current_drop_selection

#     # # nmin = np.min(images, axis=(1, 2)).reshape(-1, 1)
#     # # nmax = np.max(images, axis=(1, 2)).reshape(-1, 1)
#     # # images = images.reshape(images.shape[0], -1)
#     # # images = (images - nmin)/(nmax - nmin)
#     # images = images.reshape(images.shape[0], 50, 50, 1)

#     # for iid in range(10):
#     #     plt.pcolormesh(images[iid].reshape(50, 50))
#     #     plt.colorbar()
#     #     plt.show()

#     # print(np.min(images))
#     # print(np.max(images))

#     # print(np.min(exp_data))
#     # print(np.max(exp_data))

#     return None
