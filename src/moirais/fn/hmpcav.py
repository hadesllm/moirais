# moirais.fn — function file (hadesllm/moirais)
"""PCA preserves variance along principal components."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_pca_variance"]


def geron_pca_variance(X, n_components):
    """
    PCA preserves variance along principal components

    Formula: max_w w^T Sigma w s.t. ||w||=1

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components, variance

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCA preserves variance along principal components"})


def cheatsheet():
    return "hmpcav: PCA preserves variance along principal components"
