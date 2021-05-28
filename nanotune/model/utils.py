import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fp
import scipy.signal as sg
from scipy.ndimage import gaussian_filter, generic_gradient_magnitude, sobel
from skimage.transform import resize

import nanotune as nt
from nanotune.data.dataset import default_coord_names

N_2D = nt.config["core"]["standard_shapes"]["2"]


def generate_one_f_noise(
    how_many: int = 20000,
    save_to_file: bool = True,
    filename: Optional[str] = None,
) -> np.ndarray:
    """ """
    fx_1d = fp.frequenciesshift(fp.frequenciesfreq(1000, d=0.02))

    condensed_data_all = np.empty(
        [len(nt.config["core"]["data_types"]) - 1, 0, np.prod(N_2D)]
    )

    for niter in range(how_many):

        condensed_data = np.empty(
            [len(nt.config["core"]["data_types"]) - 1, 1, np.prod(N_2D)]
        )

        fx, fy = np.meshgrid(fx_1d, fx_1d, indexing="ij")
        f = np.sqrt(fx ** 2 + fy ** 2)

        f[f > 0] = np.divide(1, f[f > 0])

        # if low_pass_cutoff is not None:
        #     f[f > low_pass_cutoff] = 0

        # if high_pass_cutoff is not None:
        # f[f < high_pass_cutoff] = 0

        exponents = np.random.uniform(low=0, high=2 * np.pi, size=f.shape)
        power_spect = np.multiply(f, np.exp(1j * exponents))

        noise = np.abs(fp.ifrequencies2(power_spect))
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

        grad = generic_gradient_magnitude(noise, sobel)

        noise = resize(noise, N_2D, anti_aliasing=True, mode="constant").flatten()

        grad = resize(grad, N_2D, anti_aliasing=True, mode="constant").flatten()

        power_spect = resize(
            np.abs(power_spect), N_2D, anti_aliasing=True, mode="constant"
        ).flatten()

        index = nt.config["core"]["data_types"]["signal"]
        condensed_data[index, 0, :] = noise

        index = nt.config["core"]["data_types"]["frequencies"]
        condensed_data[index, 0, :] = power_spect

        index = nt.config["core"]["data_types"]["gradient"]
        condensed_data[index, 0, :] = grad

        condensed_data_all = np.concatenate(
            (condensed_data_all, condensed_data), axis=1
        )

    if save_to_file:
        if filename is None:
            filename = "one_over_f_noise.npy"
        path = os.path.join(nt.config["db_folder"], filename)
        np.save(path, condensed_data_all)

    return condensed_data_all


def generate_white_noise(
    how_many: int = 20000,
    save_to_file: bool = True,
    filename: Optional[str] = None,
) -> np.ndarray:
    """ """
    condensed_data_all = np.empty(
        [len(nt.config["core"]["data_types"]) - 1, 0, np.prod(N_2D)]
    )

    for niter in range(how_many):
        condensed_data = np.empty(
            [len(nt.config["core"]["data_types"]) - 1, 1, np.prod(N_2D)]
        )
        coeff = np.random.normal(0, 1, N_2D)
        noise = np.abs(fp.ifrequencies2(coeff))
        grad = generic_gradient_magnitude(noise, sobel)

        index = nt.config["core"]["data_types"]["signal"]
        condensed_data[index, 0, :] = noise.flatten()

        index = nt.config["core"]["data_types"]["frequencies"]
        condensed_data[index, 0, :] = coeff.flatten()

        index = nt.config["core"]["data_types"]["gradient"]
        condensed_data[index, 0, :] = grad.flatten()

        condensed_data_all = np.concatenate(
            (condensed_data_all, condensed_data), axis=1
        )

    if save_to_file:
        if filename is None:
            filename = "white_noise.npy"
        path = os.path.join(nt.config["db_folder"], filename)
        np.save(path, condensed_data_all)

    return condensed_data_all


def generate_current_drop(
    how_many: int = 20000,
    save_to_file: bool = True,
    filename: Optional[str] = None,
) -> np.ndarray:
    """ """
    condensed_data_all = np.empty(
        [len(nt.config["core"]["data_types"]) - 1, 0, np.prod(N_2D)]
    )

    for niter in range(how_many):
        condensed_data = np.empty(
            [len(nt.config["core"]["data_types"]) - 1, 1, np.prod(N_2D)]
        )
        xm, ym = np.meshgrid(np.linspace(0, 50, 50), np.linspace(0, 50, 50))
        drop = np.sqrt((xm + ym) ** 2)
        drop = (drop - np.min(drop)) / (np.max(drop) - np.min(drop))

        amp = np.random.uniform(0, 10, 1)
        offset = np.random.uniform(-5, 5, 1)

        drop = np.tanh(amp * drop + offset)
        drop = (drop - np.min(drop)) / (np.max(drop) - np.min(drop))

        drop_freq = fp.frequencies2(drop)
        drop_freq = fp.frequenciesshift(drop_freq)
        drop_freq = np.abs(drop_freq)

        grad = generic_gradient_magnitude(drop, sobel)

        index = nt.config["core"]["data_types"]["signal"]
        condensed_data[index, 0, :] = drop.flatten()

        index = nt.config["core"]["data_types"]["frequencies"]
        condensed_data[index, 0, :] = drop_freq.flatten()

        index = nt.config["core"]["data_types"]["gradient"]
        condensed_data[index, 0, :] = grad.flatten()

        condensed_data_all = np.concatenate(
            (condensed_data_all, condensed_data), axis=1
        )

    if save_to_file:
        if filename is None:
            filename = "current_drop.npy"
        path = os.path.join(nt.config["db_folder"], filename)
        np.save(path, condensed_data_all)

    return condensed_data_all


def generate_random_telegraph_noise(
    how_many: int = 20000,
    save_to_file: bool = True,
    filename: Optional[str] = None,
) -> np.ndarray:
    """ """
    condensed_data_all = np.empty(
        [len(nt.config["core"]["data_types"]) - 1, 0, np.prod(N_2D)]
    )

    for niter in range(how_many):
        condensed_data = np.empty(
            [len(nt.config["core"]["data_types"]) - 1, 1, np.prod(N_2D)]
        )
        x = np.ones(N_2D)
        s = 1
        # for n_switches in range(0, 1):

        lam = np.random.uniform(0, 0.2, 1)
        trnsp = np.random.randint(2, size=1)

        poisson = np.random.poisson(lam=lam, size=N_2D)
        poisson[poisson > 1] = 1
        for ix in range(N_2D[0]):
            for iy in range(N_2D[0]):
                if poisson[ix, iy] == 1:
                    s *= -1
                x[ix, iy] *= s
        if trnsp:
            x = x.T

        x = (x + 1) / 2

        noise_spect = fp.frequencies2(x)
        noise_spect = fp.frequenciesshift(noise_spect)
        noise_spect = np.abs(noise_spect)

        grad = generic_gradient_magnitude(x, sobel)

        index = nt.config["core"]["data_types"]["signal"]
        condensed_data[index, 0, :] = x.flatten()

        index = nt.config["core"]["data_types"]["frequencies"]
        condensed_data[index, 0, :] = noise_spect.flatten()

        index = nt.config["core"]["data_types"]["gradient"]
        condensed_data[index, 0, :] = grad.flatten()

        condensed_data_all = np.concatenate(
            (condensed_data_all, condensed_data), axis=1
        )

    if save_to_file:
        if filename is None:
            filename = "random_telegraph_noise.npy"
        path = os.path.join(nt.config["db_folder"], filename)
        np.save(path, condensed_data_all)

    return condensed_data_all


# define normalized 2D gaussian
def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    norm = 1.0 / (2.0 * np.pi * sx * sy)
    norm = norm * np.exp(
        -((x - mx) ** 2.0 / (2.0 * sx ** 2.0) + (y - my) ** 2.0 / (2.0 * sy ** 2.0))
    )
    return norm


def generate_random_blobs(
    how_many: int = 20000,
    save_to_file: bool = True,
    filename: Optional[str] = None,
    n_blobs: int = 15,
    stdx: Optional[List[float]] = None,
    stdy: Optional[List[float]] = None,
) -> np.ndarray:
    """ """
    if stdx is None:
        stdx = [0.3, 0.8]
    if stdy is None:
        stdy = [0.3, 0.8]

    condensed_data_all = np.empty(
        [len(nt.config["core"]["data_types"]) - 1, 0, np.prod(N_2D)]
    )

    for niter in range(how_many):
        condensed_data = np.empty(
            [len(nt.config["core"]["data_types"]) - 1, 1, np.prod(N_2D)]
        )
        x = np.linspace(-1, 1)
        y = np.linspace(-1, 1)
        x, y = np.meshgrid(x, y)
        z = np.zeros(N_2D)
        for n_blob in range(n_blobs):
            z += gauss2d(
                x,
                y,
                mx=np.random.uniform(-1, 1, 1),
                my=np.random.uniform(-1, 1, 1),
                sx=np.random.uniform(*stdx, 1),
                sy=np.random.uniform(*stdy, 1),
            )
        z = (z - np.min(z)) / (np.max(z) - np.min(z))

        noise_spect = fp.frequencies2(z)
        noise_spect = fp.frequenciesshift(noise_spect)
        noise_spect = np.abs(noise_spect)

        grad = generic_gradient_magnitude(z, sobel)

        index = nt.config["core"]["data_types"]["signal"]
        condensed_data[index, 0, :] = z.flatten()

        index = nt.config["core"]["data_types"]["frequencies"]
        condensed_data[index, 0, :] = noise_spect.flatten()

        index = nt.config["core"]["data_types"]["gradient"]
        condensed_data[index, 0, :] = grad.flatten()

        condensed_data_all = np.concatenate(
            (condensed_data_all, condensed_data), axis=1
        )

    if save_to_file:
        if filename is None:
            filename = "random_blobs.npy"
        path = os.path.join(nt.config["db_folder"], filename)
        np.save(path, condensed_data_all)

    return condensed_data_all
