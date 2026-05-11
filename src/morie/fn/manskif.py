"""Manski no-assumption bounds."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["manski_bounds"]


def manski_bounds(Y, X):
    """
    Manski no-assumption bounds

    Formula: width = (1 − E[obs]) for binary Y

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski (1990)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Manski no-assumption bounds"})


def cheatsheet():
    return "manskif: Manski no-assumption bounds"
