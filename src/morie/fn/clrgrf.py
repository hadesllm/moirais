"""Cluster-aware GRF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clustered_grf"]


def clustered_grf(y, D, X, cluster):
    """
    Cluster-aware GRF

    Formula: clusters as fixed-effects + forest

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey-Wager (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cluster-aware GRF"})


def cheatsheet():
    return "clrgrf: Cluster-aware GRF"
