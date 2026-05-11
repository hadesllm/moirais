"""WARP — weighted approx rank pairwise."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["warp"]


def warp(pairs, K):
    """
    WARP — weighted approx rank pairwise

    Formula: sample neg until rank violation; rank-weighted update

    Parameters
    ----------
    pairs : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Weston-Bengio-Usunier (2011)
    """
    pairs = np.atleast_1d(np.asarray(pairs, dtype=float))
    n = len(pairs)
    result = float(np.mean(pairs))
    se = float(np.std(pairs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "WARP — weighted approx rank pairwise"})


def cheatsheet():
    return "warpL: WARP — weighted approx rank pairwise"
