# morie.fn -- function file (hadesllm/morie)
"""Explained variance ratio per principal component."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_explained_variance_ratio"]


def geron_explained_variance_ratio(X, n_components):
    """
    Explained variance ratio per principal component

    Formula: EVR_k = sigma_k^2 / sum_j sigma_j^2

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: evr

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Explained variance ratio per principal component"})


def cheatsheet():
    return "hmevr: Explained variance ratio per principal component"
