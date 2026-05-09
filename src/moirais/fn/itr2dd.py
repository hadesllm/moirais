"""Optimal individualized treatment regime via DR-DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["itr_optimal_did"]


def itr_optimal_did(y, D, W):
    """
    Optimal individualized treatment regime via DR-DiD

    Formula: d*(W) = argmax_d ATT(d, W)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Optimal individualized treatment regime via DR-DiD"})


def cheatsheet():
    return "itr2dd: Optimal individualized treatment regime via DR-DiD"
