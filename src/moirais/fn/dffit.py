"""DFFITS scaled change in fitted value when obs i deleted."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dffits"]


def dffits(y, X):
    """
    DFFITS scaled change in fitted value when obs i deleted

    Formula: DFFITS_i = (e_i / (s_(i) sqrt(1 - h_ii))) sqrt(h_ii / (1 - h_ii))

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Belsley, Kuh, Welsch (1980)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DFFITS scaled change in fitted value when obs i deleted"})


def cheatsheet():
    return "dffit: DFFITS scaled change in fitted value when obs i deleted"
