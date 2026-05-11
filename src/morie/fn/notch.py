# morie.fn — function file (hadesllm/morie)
"""Notch filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def notch_filter_signal(x, freq: float, fs: float, q: float = 30.0) -> SignalResult:
    """Apply a notch (band-reject) filter to suppress a specific frequency.

    Parameters
    ----------
    x : array-like
        Input signal.
    freq : float
        Frequency to reject (Hz).
    fs : float
        Sampling frequency (Hz).
    q : float
        Quality factor controlling notch bandwidth. Default 30.0.

    Returns
    -------
    SignalResult
    """
    from morie._filters import notch_filter as _nf

    x = np.asarray(x, dtype=float)
    result = _nf(x, freq=freq, fs=fs, q=q)
    return SignalResult(name="notch_filter_signal", filtered=result, fs=fs, n_samples=len(x))


notch = notch_filter_signal


def cheatsheet() -> str:
    return "notch_filter_signal({}) -> Notch filter."
