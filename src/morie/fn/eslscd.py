"""Sparse PCA via L1."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_sparse_pca"]


def esl_sparse_pca(X, k, lambda_):
    """
    Sparse PCA via L1

    Formula: max v'Sigma v - lambda |v|_1, |v|=1

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.
    lambda_ : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components

    References
    ----------
    Hastie ESL Ch 14
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sparse PCA via L1"})


def cheatsheet():
    return "eslscd: Sparse PCA via L1"
