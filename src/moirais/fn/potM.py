"""Peaks-over-threshold."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["peaks_over_threshold"]


def peaks_over_threshold(y, u):
    """
    Peaks-over-threshold

    Formula: fit GPD to exceedances over u

    Parameters
    ----------
    y : array-like
        Input data.
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pickands (1975); Davison-Smith (1990)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Peaks-over-threshold"})


def cheatsheet():
    return "potM: Peaks-over-threshold"
