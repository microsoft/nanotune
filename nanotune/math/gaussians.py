import numpy as np
from scipy import optimize
from scipy.stats import multivariate_normal


def gaussian1D_fct(height: float, center: float, width: float):
    """"""
    width = float(width)
    return lambda x: height * np.exp(-(((center - x) / width) ** 2) / 2)


def gaussian1D_pdf(center: float, width: float):
    """"""
    rv = multivariate_normal(center, width)
    return rv.pdf


def gaussian2D_pdf(
    mean_x: float,
    mean_y: float,
    cov_xx: float,
    cov_xy: float,
    cov_yx: float,
    cov_yy: float,
):
    mean = np.array([mean_x, mean_y])
    cov = np.array([[cov_xx, cov_xy], [cov_yx, cov_yy]])
    rv = multivariate_normal(mean, cov)
    return rv.pdf


def gaussian2D_fct(
    height: float, center_x: float, center_y: float, width_x: float, width_y: float
):
    """
    Use like this:
    v_x: voltages on x axis
    v_y: votlages on y axis

    gauss = gaussian2D(1, center_gauss_x , center_gauss_y, width_x, width_y)
    Xin, Yin = np.meshgrid(v_x, v_y)

    gauss_data = gauss(Xin, Yin)
    """
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2
    )


def moments2D(data: np.ndarray):
    # """
    # Estimates initial guesses for fit paramters of a 2d gaussian.
    # Use like this:
    # step_fct_corner: 2d vector of data we want to fit

    # params = moments(step_fct_corner)
    # errorfunction = lambda p: ravel(gaussian2D(*p)(Xin, Yin) -  step_fct_corner)
    # p, success = optimize.leastsq(errorfunction, params)
    # """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y
