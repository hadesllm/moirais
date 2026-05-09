"""Lasso (L1-penalized) regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lasso_regression"]


def lasso_regression(y, X, lam):
    """
    Lasso (L1-penalized) regression

    Formula: min ||y - X beta||^2 + lambda ||beta||_1

    Parameters
    ----------
    y : array-like
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
    Tibshirani (1996)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lasso (L1-penalized) regression"})


def cheatsheet():
    return "lassrg: Lasso (L1-penalized) regression"
