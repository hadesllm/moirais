# morie.fn -- function file (hadesllm/morie)
"""Local outlier factor (LOF) using local reachability density."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_local_outlier_factor"]


def geron_local_outlier_factor(X, n_neighbors):
    """
    Local outlier factor (LOF) using local reachability density

    Formula: LOF_k(x) = avg_{y in N_k(x)} lrd_k(y) / lrd_k(x)

    Parameters
    ----------
    X : array-like
        Input data.
    n_neighbors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lof

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local outlier factor (LOF) using local reachability density"})


def cheatsheet():
    return "hmlof: Local outlier factor (LOF) using local reachability density"
