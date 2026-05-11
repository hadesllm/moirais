# morie.fn — function file (hadesllm/morie)
"""Moving average filter."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Rebellions are built on hope."


def moving_average(x, window=5, **kwargs) -> DescriptiveResult:
    """Apply a simple moving average filter.

    .. math::

        y(n) = \\frac{1}{M} \\sum_{k=0}^{M-1} x(n - k)

    Parameters
    ----------
    x : array-like
        Input signal.
    window : int
        Filter window length M.

    Returns
    -------
    DescriptiveResult
        ``value`` is the smoothed signal (ndarray).
    """
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(x, kernel, mode="same")
    return DescriptiveResult(
        name="moving_average",
        value=smoothed,
        extra={"window": window, "n": len(x)},
    )


mvavg = moving_average


def cheatsheet() -> str:
    return "moving_average({}) -> Moving average filter."
