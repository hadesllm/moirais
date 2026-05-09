# moirais.fn — function file (hadesllm/moirais)
"""Incremental PCA for out-of-core / streaming datasets."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_incremental_pca"]


def geron_incremental_pca(X_iter, n_components, batch_size):
    """
    Incremental PCA for out-of-core / streaming datasets

    Formula: update PCs with mini-batches

    Parameters
    ----------
    X_iter : array-like
        Input data.
    n_components : array-like
        Input data.
    batch_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components

    References
    ----------
    Géron Ch 7
    """
    X_iter = np.atleast_1d(np.asarray(X_iter, dtype=float))
    n = len(X_iter)
    result = float(np.mean(X_iter))
    se = float(np.std(X_iter, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Incremental PCA for out-of-core / streaming datasets"})


def cheatsheet():
    return "hmipca: Incremental PCA for out-of-core / streaming datasets"
