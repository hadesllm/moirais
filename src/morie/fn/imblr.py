# morie.fn — function file (hadesllm/morie)
"""Look well into thyself; there is a source which will always spring up. — Marcus Aurelius"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ._containers import DescriptiveResult


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> DescriptiveResult:
    r"""
    Apply Gaussian blur to a grayscale image.

    Convolves the image with a 2-D Gaussian kernel of standard deviation
    *sigma*:

    .. math::

        G(x, y) = \\frac{1}{2\\pi\\sigma^2}
                   \\exp\\!\\left(-\\frac{x^2 + y^2}{2\\sigma^2}\\right)

    :param image: 2-D grayscale image array.
    :type image: numpy.ndarray
    :param sigma: Standard deviation of the Gaussian kernel. Default 1.0.
    :type sigma: float
    :return: DescriptiveResult with blurred image in extra.
    :rtype: DescriptiveResult
    :raises ValueError: If image is not 2-D or sigma <= 0.

    References
    ----------
    Gonzalez R.C. & Woods R.E. (2018). *Digital Image Processing*,
    4th ed. Pearson. Chapter 3.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got {image.ndim}-D.")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    blurred = gaussian_filter(image, sigma=sigma)
    return DescriptiveResult(
        name="gaussian_blur",
        value=float(sigma),
        extra={"blurred": blurred, "sigma": sigma},
    )


imblr = gaussian_blur


def cheatsheet() -> str:
    return "gaussian_blur({}) -> Gaussian blur via convolution. 'In my experience there is no"
