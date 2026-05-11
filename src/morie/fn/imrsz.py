# morie.fn — function file (hadesllm/morie)
"""It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def image_resize(image: np.ndarray, scale: float = 0.5) -> DescriptiveResult:
    """
    Resize a grayscale image using bilinear interpolation.

    Maps each output pixel to the input coordinate space and computes
    the weighted average of the four nearest input pixels.

    :param image: 2-D grayscale image array.
    :type image: numpy.ndarray
    :param scale: Scale factor (e.g. 0.5 halves dimensions). Must be > 0.
    :type scale: float
    :return: DescriptiveResult with resized image in extra.
    :rtype: DescriptiveResult
    :raises ValueError: If image is not 2-D or scale <= 0.

    References
    ----------
    Gonzalez R.C. & Woods R.E. (2018). *Digital Image Processing*,
    4th ed. Pearson. Chapter 2.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got {image.ndim}-D.")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    h, w = image.shape
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    row_idx = np.linspace(0, h - 1, new_h)
    col_idx = np.linspace(0, w - 1, new_w)
    r, c = np.meshgrid(row_idx, col_idx, indexing="ij")
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    dr = r - r0
    dc = c - c0
    resized = (
        image[r0, c0] * (1 - dr) * (1 - dc)
        + image[r1, c0] * dr * (1 - dc)
        + image[r0, c1] * (1 - dr) * dc
        + image[r1, c1] * dr * dc
    )
    return DescriptiveResult(
        name="image_resize",
        value=scale,
        extra={"resized": resized, "original_shape": (h, w), "new_shape": (new_h, new_w)},
    )


imrsz = image_resize


def cheatsheet() -> str:
    return "image_resize({}) -> Bilinear interpolation image resize. 'Never tell me the odds"
