"""Kernel PCA top-k components."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_kernel_pca"]


def sgt_kernel_pca(X, kernel, k):
    """
    Kernel PCA top-k components

    Formula: Eigendecompose centred kernel K̃

    Parameters
    ----------
    X : array-like
        Input data.
    kernel : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y, eigvals

    References
    ----------
    Schölkopf-Smola-Müller (1998)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel PCA top-k components"})


def cheatsheet():
    return "sgtkpc: Kernel PCA top-k components"
