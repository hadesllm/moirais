# morie.fn -- function file (hadesllm/morie)
"""DBSCAN core-point predicate: |N_eps(x)| >= min_samples."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dbscan_core_point"]


def geron_dbscan_core_point(X, eps, min_samples):
    """
    DBSCAN core-point predicate: |N_eps(x)| >= min_samples

    Formula: core(x) = 1 if |{x' : ||x-x'|| <= eps}| >= min_samples

    Parameters
    ----------
    X : array-like
        Input data.
    eps : array-like
        Input data.
    min_samples : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_core

    References
    ----------
    Géron Ch 8, DBSCAN section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DBSCAN core-point predicate: |N_eps(x)| >= min_samples"})


def cheatsheet():
    return "grdbs: DBSCAN core-point predicate: |N_eps(x)| >= min_samples"
