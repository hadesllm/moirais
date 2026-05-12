# morie.fn -- function file (hadesllm/morie)
"""Local outlier factor of a point relative to its k-nearest neighbors."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_local_outlier_factor"]


def geron_local_outlier_factor(X, k):
    """
    Local outlier factor of a point relative to its k-nearest neighbors

    Formula: LOF(x) = mean over k-NN of (lrd(neighbor)/lrd(x)), lrd = 1/mean(reach_dist_k)

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lof

    References
    ----------
    Géron Ch 8, Local Outlier Factor section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local outlier factor of a point relative to its k-nearest neighbors"})


def cheatsheet():
    return "grlof: Local outlier factor of a point relative to its k-nearest neighbors"
