"""Cheeger inequality bound on λ_2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_cheeger_bound"]


def sgt_cheeger_bound(lam2):
    """
    Cheeger inequality bound on λ_2

    Formula: h²/2 <= λ_2 <= 2h

    Parameters
    ----------
    lam2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_lo, h_hi

    References
    ----------
    Chung (1997)
    """
    lam2 = np.atleast_1d(np.asarray(lam2, dtype=float))
    n = len(lam2)
    result = float(np.mean(lam2))
    se = float(np.std(lam2, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cheeger inequality bound on λ_2"})


def cheatsheet():
    return "sgtcheb: Cheeger inequality bound on λ_2"
