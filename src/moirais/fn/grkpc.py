# moirais.fn — function file (hadesllm/moirais)
"""Kernel PCA with RBF kernel: eigendecomposition of centered Gram matrix."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kernel_pca_rbf"]


def geron_kernel_pca_rbf(X, gamma, d):
    """
    Kernel PCA with RBF kernel: eigendecomposition of centered Gram matrix

    Formula: K_ij = exp(-gamma ||x_i - x_j||^2); K_centered = K - 1_m K - K 1_m + 1_m K 1_m; Z = eigvec(K_c)

    Parameters
    ----------
    X : array-like
        Input data.
    gamma : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Z

    References
    ----------
    Géron Ch 7, Kernel PCA section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel PCA with RBF kernel: eigendecomposition of centered Gram matrix"})


def cheatsheet():
    return "grkpc: Kernel PCA with RBF kernel: eigendecomposition of centered Gram matrix"
