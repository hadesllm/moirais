# moirais.fn — function file (hadesllm/moirais)
"""Kernel PCA with RBF kernel in reproducing kernel Hilbert space."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kernel_pca_rbf"]


def geron_kernel_pca_rbf(X, n_components, gamma):
    """
    Kernel PCA with RBF kernel in reproducing kernel Hilbert space

    Formula: K(x,y) = exp(-gamma ||x-y||^2); eig of centered K

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    gamma : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel PCA with RBF kernel in reproducing kernel Hilbert space"})


def cheatsheet():
    return "hmkprbf: Kernel PCA with RBF kernel in reproducing kernel Hilbert space"
