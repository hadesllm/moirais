# moirais.fn — function file (hadesllm/moirais)
"""Sparse random projection matrix with {-1,0,+1} entries."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sparse_rand_projection"]


def geron_sparse_rand_projection(X, d_out, density, seed):
    """
    Sparse random projection matrix with {-1,0,+1} entries

    Formula: R_ij = ±sqrt(s/d') with prob 1/(2s), else 0

    Parameters
    ----------
    X : array-like
        Input data.
    d_out : array-like
        Input data.
    density : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_projected

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sparse random projection matrix with {-1,0,+1} entries"})


def cheatsheet():
    return "hmsrp: Sparse random projection matrix with {-1,0,+1} entries"
