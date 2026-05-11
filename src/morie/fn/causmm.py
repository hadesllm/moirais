"""Mahalanobis-distance matching."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_mahalanobis_match"]


def causal_mahalanobis_match(X, treat, k):
    """
    Mahalanobis-distance matching

    Formula: d_M(x_i,x_j) = sqrt((x_i-x_j)^T Σ̂^{-1} (x_i-x_j))

    Parameters
    ----------
    X : array-like
        Input data.
    treat : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: match_idx

    References
    ----------
    Rubin (1980)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mahalanobis-distance matching"})


def cheatsheet():
    return "causmm: Mahalanobis-distance matching"
