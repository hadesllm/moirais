# morie.fn — function file (hadesllm/morie)
"""Limits of k-means: fails with non-spherical or varying-size clusters."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kmeans_limits"]


def geron_kmeans_limits(X):
    """
    Limits of k-means: fails with non-spherical or varying-size clusters

    Formula: assumes isotropic clusters; fails for anisotropic

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: limitations

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Limits of k-means: fails with non-spherical or varying-size clusters"})


def cheatsheet():
    return "hmkmlim: Limits of k-means: fails with non-spherical or varying-size clusters"
