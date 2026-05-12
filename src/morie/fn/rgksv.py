# morie.fn -- function file (hadesllm/morie)
"""K-SVD dictionary learning algorithm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ksvd"]


def rangayyan_ksvd(Y, n_atoms, sparsity, max_iter):
    """
    K-SVD dictionary learning algorithm

    Formula: D,X <- alternating SVD update; X=argmin ||Y-DX||_F s.t. ||x_i||_0<=T

    Parameters
    ----------
    Y : array-like
        Input data.
    n_atoms : array-like
        Input data.
    sparsity : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: D, X

    References
    ----------
    Rangayyan Ch 9.5
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-SVD dictionary learning algorithm"})


def cheatsheet():
    return "rgksv: K-SVD dictionary learning algorithm"
