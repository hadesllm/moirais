"""Weighted frequency table."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weighted_frequency"]


def weighted_frequency(y, weights, cells):
    """
    Weighted frequency table

    Formula: f_k = sum_{i in cell k} w_i

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    cells : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lohr (2010) §7
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weighted frequency table"})


def cheatsheet():
    return "wfrep: Weighted frequency table"
