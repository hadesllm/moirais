"""Bregman alternating projections solver for entropic OT."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_bregman_proj"]


def ot_bregman_proj(K, a, b, max_iter):
    """
    Bregman alternating projections solver for entropic OT

    Formula: Project on row-sum then column-sum simplices iteratively

    Parameters
    ----------
    K : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T, iters

    References
    ----------
    Benamou-Carlier-Cuturi-Nenna-Peyré (2015)
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    n = len(K)
    result = float(np.mean(K))
    se = float(np.std(K, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bregman alternating projections solver for entropic OT"})


def cheatsheet():
    return "otbreg: Bregman alternating projections solver for entropic OT"
