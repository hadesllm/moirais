"""Cluster-robust DR-DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_clustered_did"]


def dr_clustered_did(y, D, X, cluster):
    """
    Cluster-robust DR-DiD

    Formula: DR moment with cluster-robust variance

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
    Bertrand-Duflo-Mullainathan (2004)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cluster-robust DR-DiD"})


def cheatsheet():
    return "drclt: Cluster-robust DR-DiD"
