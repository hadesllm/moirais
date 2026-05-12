# morie.fn -- function file (hadesllm/morie)
"""Median filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def median_filter_signal(x, kernel_size: int = 5) -> SignalResult:
    """Apply a median filter to remove impulse noise from signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.
    kernel_size : int
        Size of the median kernel (must be odd). Default 5.

    Returns
    -------
    SignalResult
    """
    from morie._filters import median_filter as _mdf

    x = np.asarray(x, dtype=float)
    result = _mdf(x, kernel_size=kernel_size)
    return SignalResult(name="median_filter_signal", filtered=result, fs=0.0, n_samples=len(x))


medf = median_filter_signal


def cheatsheet() -> str:
    return "median_filter_signal({}) -> Median filter."
