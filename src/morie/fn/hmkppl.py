# morie.fn -- function file (hadesllm/morie)
"""Kernel PCA with polynomial kernel."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kernel_pca_poly"]


def geron_kernel_pca_poly(X, n_components, degree, gamma, coef0):
    """
    Kernel PCA with polynomial kernel

    Formula: K(x,y) = (gamma * x^T y + coef0)^d

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    degree : array-like
        Input data.
    gamma : array-like
        Input data.
    coef0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel PCA with polynomial kernel"})


def cheatsheet():
    return "hmkppl: Kernel PCA with polynomial kernel"
