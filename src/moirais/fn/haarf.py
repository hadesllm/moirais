# moirais.fn — function file (hadesllm/moirais)
"""Haar wavelet transform."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Character is destiny. — Heraclitus"


def haar_transform(x: np.ndarray) -> DescriptiveResult:
    """Haar wavelet transform (simplest DWT, db1).

    The Haar wavelet uses averaging and differencing of adjacent samples:

    .. math::

        c(n) = \\frac{x(2n) + x(2n+1)}{\\sqrt{2}}, \\quad
        d(n) = \\frac{x(2n) - x(2n+1)}{\\sqrt{2}}

    Parameters
    ----------
    x : array-like
        1-D input signal (length should be even; padded with 0 if odd).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``coeffs`` [cA, cD], ``approximation``, ``detail``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) % 2 != 0:
        x = np.append(x, 0.0)
    s2 = np.sqrt(2)
    approx = (x[0::2] + x[1::2]) / s2
    detail = (x[0::2] - x[1::2]) / s2
    return DescriptiveResult(
        name="haar_transform",
        value=float(len(approx)),
        extra={"coeffs": [approx, detail], "approximation": approx, "detail": detail},
    )


haarf = haar_transform


def cheatsheet() -> str:
    return "haar_transform({}) -> Haar wavelet transform."
