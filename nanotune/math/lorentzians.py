def lorentzian_1D(x, amplitude, x_0, fwhm):
    """One dimensional Lorentzian model function"""

    numerator = (fwhm / 2.0) ** 2
    denominator = (x - x_0) ** 2 + (fwhm / 2.0) ** 2
    lor = numerator / denominator
    lor = amplitude * lor
    return lor
