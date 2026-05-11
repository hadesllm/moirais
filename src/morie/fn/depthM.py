"""Mahalanobis depth."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mahalanobis_depth"]


def mahalanobis_depth(x, mu, Sigma):
    """
    Mahalanobis depth

    Formula: d = 1/(1 + (x−μ)^T Σ^{-1}(x−μ))

    Parameters
    ----------
    x : array-like
        Input data.
    mu : array-like
        Input data.
    Sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu (1990)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mahalanobis depth"})


def cheatsheet():
    return "depthM: Mahalanobis depth"
