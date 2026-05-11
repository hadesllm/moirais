# morie.fn — function file (hadesllm/morie)
"""Homomorphic filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def homomorphic_filter(x, cutoff: float = 0.1, fs: float = 1.0) -> SignalResult:
    """Apply a homomorphic filter to signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff : float
        Cutoff frequency. Default 0.1.
    fs : float
        Sampling frequency. Default 1.0.

    Returns
    -------
    SignalResult
    """
    from morie._detection import homomorphic_filter as _hm

    x = np.asarray(x, dtype=float)
    result = _hm(x, cutoff=cutoff, fs=fs)
    return SignalResult(name="homomorphic_filter", filtered=result, fs=fs, n_samples=len(x), extra={"cutoff": cutoff})


hmflt = homomorphic_filter


def cheatsheet() -> str:
    return "homomorphic_filter({}) -> Homomorphic filter."
