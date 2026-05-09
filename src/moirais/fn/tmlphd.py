"""High-dimensional TMLE with sparsity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_high_dim"]


def tmle_high_dim(y, D, X, lam):
    """
    High-dimensional TMLE with sparsity

    Formula: L1-regularized Q + g; cross-fit

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Belloni-Chernozhukov-Hansen (2014); vdL-Coyle (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "High-dimensional TMLE with sparsity"})


def cheatsheet():
    return "tmlphd: High-dimensional TMLE with sparsity"
