"""Individualized treatment rule forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["itr_forest"]


def itr_forest(y, D, W):
    """
    Individualized treatment rule forest

    Formula: forest split criterion = -E[Y(d(W))]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey-Imbens (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Individualized treatment rule forest"})


def cheatsheet():
    return "itrgrf: Individualized treatment rule forest"
