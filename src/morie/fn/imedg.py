# morie.fn — function file (hadesllm/morie)
"""Edge detection via Sobel or Prewitt operators. 'Your focus determines your reality.' -- Qui-Gon Jinn"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from ._containers import DescriptiveResult

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
_PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)


def edge_detect(image: np.ndarray, method: str = "sobel") -> DescriptiveResult:
    r"""
    Detect edges in a grayscale image using Sobel or Prewitt operators.

    Computes horizontal and vertical gradients :math:`G_x, G_y` then
    returns the gradient magnitude :math:`|G| = \\sqrt{G_x^2 + G_y^2}`.

    :param image: 2-D grayscale image array.
    :type image: numpy.ndarray
    :param method: ``"sobel"`` or ``"prewitt"``. Default ``"sobel"``.
    :type method: str
    :return: DescriptiveResult with edge magnitude image in extra.
    :rtype: DescriptiveResult
    :raises ValueError: If image is not 2-D or method is unknown.

    References
    ----------
    Sobel I. (1968). An Isotropic 3x3 Image Gradient Operator.
    Prewitt J.M.S. (1970). *Object Enhancement and Extraction*.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got {image.ndim}-D.")
    method = method.lower()
    if method == "sobel":
        kx, ky = _SOBEL_X, _SOBEL_Y
    elif method == "prewitt":
        kx, ky = _PREWITT_X, _PREWITT_Y
    else:
        raise ValueError(f"Unknown method '{method}'; use 'sobel' or 'prewitt'.")
    gx = convolve(image, kx)
    gy = convolve(image, ky)
    mag = np.sqrt(gx**2 + gy**2)
    return DescriptiveResult(
        name="edge_detect",
        value=float(np.mean(mag)),
        extra={"magnitude": mag, "gx": gx, "gy": gy, "method": method},
    )


imedg = edge_detect


def cheatsheet() -> str:
    return "edge_detect({}) -> Edge detection via Sobel or Prewitt operators. 'Your focus d"
