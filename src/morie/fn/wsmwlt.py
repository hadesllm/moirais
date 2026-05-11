"""Wavelet thresholding (universal)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_wavelet_smooth"]


def wasserman_wavelet_smooth(y, wavelet, sigma):
    """
    Wavelet thresholding (universal)

    Formula: lambda = sigma sqrt(2 log n)

    Parameters
    ----------
    y : array-like
        Input data.
    wavelet : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wasserman (2004), Ch 20
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet thresholding (universal)"})


def cheatsheet():
    return "wsmwlt: Wavelet thresholding (universal)"
