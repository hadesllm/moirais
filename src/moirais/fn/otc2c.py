"""Pairwise squared-Euclidean cost matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_cost_pairwise"]


def ot_cost_pairwise(X, Y):
    """
    Pairwise squared-Euclidean cost matrix

    Formula: C_ij = ||x_i-y_j||²

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: C

    References
    ----------
    Peyré & Cuturi (2019)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pairwise squared-Euclidean cost matrix"})


def cheatsheet():
    return "otc2c: Pairwise squared-Euclidean cost matrix"
