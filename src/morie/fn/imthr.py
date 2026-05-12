# morie.fn -- function file (hadesllm/morie)
"""Image thresholding (Otsu / fixed). 'Time discovers truth. -- Seneca'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def threshold(image: np.ndarray, method: str = "otsu", value: float = 128.0) -> DescriptiveResult:
    r"""
    Binarise a grayscale image using Otsu's method or a fixed threshold.

    Otsu's method minimises intra-class variance to find an optimal
    threshold automatically:

    .. math::

        t^* = \\arg\\min_t \\; \\omega_0(t)\\sigma_0^2(t) + \\omega_1(t)\\sigma_1^2(t)

    :param image: 2-D grayscale image array.
    :type image: numpy.ndarray
    :param method: ``"otsu"`` or ``"fixed"``. Default ``"otsu"``.
    :type method: str
    :param value: Threshold value when method is ``"fixed"``. Default 128.
    :type value: float
    :return: DescriptiveResult with binary image and threshold in extra.
    :rtype: DescriptiveResult
    :raises ValueError: If image is not 2-D.

    References
    ----------
    Otsu N. (1979). A Threshold Selection Method from Gray-Level
    Histograms. *IEEE Trans. Systems, Man, and Cybernetics*, 9(1), 62-66.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got {image.ndim}-D.")
    method = method.lower()
    if method == "otsu":
        counts, edges = np.histogram(image.ravel(), bins=256)
        centres = (edges[:-1] + edges[1:]) / 2.0
        total = counts.sum()
        best_t, best_var = 0.0, -1.0
        w0, sum0 = 0.0, 0.0
        total_sum = float(np.dot(counts, centres))
        for i in range(256):
            w0 += counts[i]
            if w0 == 0:
                continue
            w1 = total - w0
            if w1 == 0:
                break
            sum0 += counts[i] * centres[i]
            m0 = sum0 / w0
            m1 = (total_sum - sum0) / w1
            between = w0 * w1 * (m0 - m1) ** 2
            if between > best_var:
                best_var = between
                best_t = centres[i]
        t = best_t
    else:
        t = float(value)
    binary = (image >= t).astype(float)
    return DescriptiveResult(
        name="threshold",
        value=t,
        extra={"binary": binary, "threshold": t, "method": method},
    )


imthr = threshold


def cheatsheet() -> str:
    return "threshold({}) -> Image thresholding (Otsu / fixed). 'Do or do not. There is n"
