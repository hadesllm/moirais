"""Transfer function frequency response."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The greatest teacher, failure is."


def transfer_function(b, a, worN: int = 512) -> DescriptiveResult:
    r"""Compute frequency response H(omega) = B(omega)/A(omega).

    .. math::

        H(e^{j\\omega}) = \\frac{\\sum_{k=0}^{M} b_k e^{-j\\omega k}}
                          {\\sum_{k=0}^{N} a_k e^{-j\\omega k}}

    Parameters
    ----------
    b : array-like
        Numerator coefficients.
    a : array-like
        Denominator coefficients.
    worN : int
        Number of frequency points. Default 512.

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal import freqz

    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    w, h = freqz(b, a, worN=worN)
    mag = np.abs(h)
    phase = np.unwrap(np.angle(h))
    return DescriptiveResult(
        name="transfer_function",
        value=float(np.max(mag)),
        extra={"frequencies": w, "magnitude": mag, "phase": phase, "H": h},
    )


trfnc = transfer_function


def cheatsheet() -> str:
    return "transfer_function({}) -> Transfer function frequency response."
