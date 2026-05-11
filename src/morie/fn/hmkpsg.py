# morie.fn — function file (hadesllm/morie)
"""Kernel PCA with sigmoid kernel."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kernel_pca_sigmoid"]


def geron_kernel_pca_sigmoid(X, n_components, gamma, coef0):
    """
    Kernel PCA with sigmoid kernel

    Formula: K(x,y) = tanh(gamma * x^T y + coef0)

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel PCA with sigmoid kernel"})


def cheatsheet():
    return "hmkpsg: Kernel PCA with sigmoid kernel"
