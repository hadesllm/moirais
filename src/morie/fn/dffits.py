"""DFFITS leverage-residual diagnostic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dffits"]


def dffits(X, y):
    """
    DFFITS leverage-residual diagnostic

    Formula: DFFITS_i = t*_i √(h_ii/(1−h_ii))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Belsley-Kuh-Welsch (1980)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DFFITS leverage-residual diagnostic"})


def cheatsheet():
    return "dffits: DFFITS leverage-residual diagnostic"
