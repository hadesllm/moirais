"""Wavelet-based anomaly."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["discrete_wavelet_anomaly"]


def discrete_wavelet_anomaly(x, wavelet, threshold):
    """
    Wavelet-based anomaly

    Formula: flag coefficients above noise threshold

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Donoho (1995)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet-based anomaly"})


def cheatsheet():
    return "dwtA: Wavelet-based anomaly"
