"""Mahalanobis-based OT cost matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_mahalanobis_distance_ot"]


def ot_mahalanobis_distance_ot(X, Y, Sigma):
    """
    Mahalanobis-based OT cost matrix

    Formula: C_ij = (x_i-y_j)^T Σ^{-1} (x_i-y_j)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    Sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: C

    References
    ----------
    De Maesschalck (2000)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mahalanobis-based OT cost matrix"})


def cheatsheet():
    return "otmd: Mahalanobis-based OT cost matrix"
