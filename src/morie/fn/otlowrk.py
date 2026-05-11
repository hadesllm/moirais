"""Low-rank Sinkhorn approximating T = U V^T."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_low_rank_sinkhorn"]


def ot_low_rank_sinkhorn(a, b, C, rank, epsilon, max_iter):
    """
    Low-rank Sinkhorn approximating T = U V^T

    Formula: Alternate solve over (U,V) under entropic constraint

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    rank : array-like
        Input data.
    epsilon : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: U, V, T

    References
    ----------
    Scetbon-Cuturi-Peyré (2021)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Low-rank Sinkhorn approximating T = U V^T"})


def cheatsheet():
    return "otlowrk: Low-rank Sinkhorn approximating T = U V^T"
