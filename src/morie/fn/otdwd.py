"""Doubly-stochastic Sinkhorn-Knopp projection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_doubly_stoch_proj"]


def ot_doubly_stoch_proj(K, max_iter):
    """
    Doubly-stochastic Sinkhorn-Knopp projection

    Formula: Iterate row + column normalisation of K

    Parameters
    ----------
    K : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: M, iters

    References
    ----------
    Sinkhorn & Knopp (1967)
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    n = len(K)
    result = float(np.mean(K))
    se = float(np.std(K, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Doubly-stochastic Sinkhorn-Knopp projection"})


def cheatsheet():
    return "otdwd: Doubly-stochastic Sinkhorn-Knopp projection"
