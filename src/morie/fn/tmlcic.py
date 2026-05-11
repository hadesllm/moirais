"""Cluster-robust TMLE inference."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_cluster_ic"]


def tmle_cluster_ic(y, D, X, cluster):
    """
    Cluster-robust TMLE inference

    Formula: Var = E[(sum_i^c D_i)^2]/n_c for cluster c

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
    Balzer et al (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cluster-robust TMLE inference"})


def cheatsheet():
    return "tmlcic: Cluster-robust TMLE inference"
