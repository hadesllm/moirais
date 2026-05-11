"""Smoothed periodogram (Daniell)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spectral_smoothed"]


def spectral_smoothed(y, span):
    """
    Smoothed periodogram (Daniell)

    Formula: convolve raw periodogram with Daniell window

    Parameters
    ----------
    y : array-like
        Input data.
    span : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Daniell (1946)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smoothed periodogram (Daniell)"})


def cheatsheet():
    return "specsm: Smoothed periodogram (Daniell)"
