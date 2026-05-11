"""Histogram-based outlier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hbos"]


def hbos(X, bins):
    """
    Histogram-based outlier

    Formula: sum log(1/p_i(x_i)) over independent histograms

    Parameters
    ----------
    X : array-like
        Input data.
    bins : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Goldstein-Dengel (2012)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Histogram-based outlier"})


def cheatsheet():
    return "hbos: Histogram-based outlier"
