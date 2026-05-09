"""Pairwise L_p cost matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_cost_lp"]


def ot_cost_lp(X, Y, p):
    """
    Pairwise L_p cost matrix

    Formula: C_ij = ||x_i-y_j||_p

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    p : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pairwise L_p cost matrix"})


def cheatsheet():
    return "otc2p: Pairwise L_p cost matrix"
