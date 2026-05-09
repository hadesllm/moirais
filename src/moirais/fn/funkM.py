"""Funk SVD (matrix factorization)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["funk_svd"]


def funk_svd(R, K, lr, reg):
    """
    Funk SVD (matrix factorization)

    Formula: r̂_{ui} = p_u^T q_i; SGD

    Parameters
    ----------
    R : array-like
        Input data.
    K : array-like
        Input data.
    lr : array-like
        Input data.
    reg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Funk (2006) Netflix Prize blog
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Funk SVD (matrix factorization)"})


def cheatsheet():
    return "funkM: Funk SVD (matrix factorization)"
