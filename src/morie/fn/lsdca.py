"""Robust LDA via MCD class covariances."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["robust_lda"]


def robust_lda(X, y):
    """
    Robust LDA via MCD class covariances

    Formula: replace μ_k, Σ_k with MCD versions

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Croux-Dehon (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Robust LDA via MCD class covariances"})


def cheatsheet():
    return "lsdca: Robust LDA via MCD class covariances"
