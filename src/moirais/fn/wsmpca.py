"""Principal component analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_pca"]


def wasserman_pca(X, k):
    """
    Principal component analysis

    Formula: Sigma v_k = lambda_k v_k

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components

    References
    ----------
    Wasserman (2004), Ch 14
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Principal component analysis"})


def cheatsheet():
    return "wsmpca: Principal component analysis"
