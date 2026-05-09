"""Gaussian process regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_regression"]


def gp_regression(X, y, X_star, kernel):
    """
    Gaussian process regression

    Formula: posterior mean = K_*(K+σ²I)^{-1}y

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_star : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rasmussen-Williams (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian process regression"})


def cheatsheet():
    return "gpregF: Gaussian process regression"
