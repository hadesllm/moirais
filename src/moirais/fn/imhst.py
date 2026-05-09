# moirais.fn — function file (hadesllm/moirais)
"""Image histogram computation. 'The greatest teacher, failure is.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def image_histogram(image: np.ndarray, bins: int = 256) -> DescriptiveResult:
    """
    Compute histogram of a grayscale image.

    Counts pixel intensities across *bins* equally spaced intervals
    spanning the value range of the image.

    :param image: 2-D array of pixel intensities (grayscale).
    :type image: numpy.ndarray
    :param bins: Number of histogram bins. Default 256.
    :type bins: int
    :return: DescriptiveResult with histogram counts and bin edges.
    :rtype: DescriptiveResult
    :raises ValueError: If image is not 2-D.

    References
    ----------
    Gonzalez R.C. & Woods R.E. (2018). *Digital Image Processing*,
    4th ed. Pearson. Chapter 3.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got {image.ndim}-D.")
    counts, edges = np.histogram(image.ravel(), bins=bins)
    return DescriptiveResult(
        name="image_histogram",
        value=None,
        extra={"counts": counts, "bin_edges": edges, "bins": bins},
    )


imhst = image_histogram


def cheatsheet() -> str:
    return "image_histogram({}) -> Image histogram computation. 'The greatest teacher, failure "
