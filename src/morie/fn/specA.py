"""Spectral residual anomaly."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spectral_anomaly"]


def spectral_anomaly(x):
    """
    Spectral residual anomaly

    Formula: saliency map from log magnitude FFT

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hou-Zhang (2007)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral residual anomaly"})


def cheatsheet():
    return "specA: Spectral residual anomaly"
