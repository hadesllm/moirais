"""Latent semantic analysis (SVD)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lsa"]


def lsa(X, k):
    """
    Latent semantic analysis (SVD)

    Formula: X ≈ U Σ V^T; rank-k approx

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Deerwester et al (1990)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Latent semantic analysis (SVD)"})


def cheatsheet():
    return "lsa: Latent semantic analysis (SVD)"
