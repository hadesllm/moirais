# morie.fn -- function file (hadesllm/morie)
"""Magnitude spectrum."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def magnitude_spectrum(x, **kwargs) -> DescriptiveResult:
    r"""Compute the magnitude spectrum |X(k)| of signal *x*.

    .. math::

        |X(k)| = \\left| \\text{FFT}\\{x\\} \\right|

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    X = np.fft.fft(x)
    mag = np.abs(X)
    return DescriptiveResult(
        name="magnitude_spectrum",
        value=float(np.max(mag)),
        extra={"magnitude": mag, "N": len(x)},
    )


magsp = magnitude_spectrum


def cheatsheet() -> str:
    return "magnitude_spectrum({}) -> Magnitude spectrum."
