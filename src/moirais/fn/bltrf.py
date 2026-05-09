# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bilinear transform (analog to digital)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"


def bilinear_transform(b_s, a_s, fs) -> DescriptiveResult:
    """Convert analog filter (s-domain) to digital (z-domain) via bilinear transform.

    .. math::

        s = \\frac{2 f_s (z - 1)}{z + 1}

    Parameters
    ----------
    b_s : array-like
        Analog numerator coefficients (descending powers of s).
    a_s : array-like
        Analog denominator coefficients (descending powers of s).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal import bilinear

    b_s = np.asarray(b_s, dtype=float)
    a_s = np.asarray(a_s, dtype=float)
    b_d, a_d = bilinear(b_s, a_s, fs)
    return DescriptiveResult(
        name="bilinear_transform",
        value=float(fs),
        extra={"b_digital": b_d, "a_digital": a_d, "b_analog": b_s, "a_analog": a_s},
    )


bltrf = bilinear_transform


def cheatsheet() -> str:
    return "bilinear_transform({}) -> Bilinear transform (analog to digital)."
