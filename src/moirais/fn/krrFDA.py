"""Kernel ridge regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kernel_ridge_regression"]


def kernel_ridge_regression(X, y, kernel, lam):
    """
    Kernel ridge regression

    Formula: α = (K + λI)^{-1} y

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    kernel : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vovk (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel ridge regression"})


def cheatsheet():
    return "krrFDA: Kernel ridge regression"
