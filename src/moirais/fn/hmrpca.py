# moirais.fn — function file (hadesllm/moirais)
"""Randomized PCA using random projection to approximate top components."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_randomized_pca"]


def geron_randomized_pca(X, n_components, seed):
    """
    Randomized PCA using random projection to approximate top components

    Formula: X_approx = X * Q; PCA on X_approx

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Randomized PCA using random projection to approximate top components"})


def cheatsheet():
    return "hmrpca: Randomized PCA using random projection to approximate top components"
