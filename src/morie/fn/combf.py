# morie.fn — function file (hadesllm/morie)
"""Comb filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def comb_filter_signal(x, fundamental: float, fs: float, n_harmonics: int = 5, q: float = 30.0) -> SignalResult:
    """Apply a comb filter to suppress a fundamental frequency and its harmonics.

    Parameters
    ----------
    x : array-like
        Input signal.
    fundamental : float
        Fundamental frequency to reject (Hz).
    fs : float
        Sampling frequency (Hz).
    n_harmonics : int
        Number of harmonics to notch. Default 5.
    q : float
        Quality factor for each notch. Default 30.0.

    Returns
    -------
    SignalResult
    """
    from morie._filters import comb_filter as _cf

    x = np.asarray(x, dtype=float)
    result = _cf(x, fundamental=fundamental, fs=fs, n_harmonics=n_harmonics, q=q)
    return SignalResult(name="comb_filter_signal", filtered=result, fs=fs, n_samples=len(x))


combf = comb_filter_signal


def cheatsheet() -> str:
    return "comb_filter_signal({}) -> Comb filter."
