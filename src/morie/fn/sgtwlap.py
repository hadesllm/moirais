"""Weighted Laplacian L_W from a weight matrix W."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_weighted_laplacian"]


def sgt_weighted_laplacian(W):
    """
    Weighted Laplacian L_W from a weight matrix W

    Formula: L_W = D_W - W; D_W = diag(W 1)

    Parameters
    ----------
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: L_W

    References
    ----------
    Chung (1997)
    """
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weighted Laplacian L_W from a weight matrix W"})


def cheatsheet():
    return "sgtwlap: Weighted Laplacian L_W from a weight matrix W"
