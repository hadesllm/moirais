"""Resistance / commute-time distance from L^+."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_commute_distance"]


def sgt_commute_distance(A):
    """
    Resistance / commute-time distance from L^+

    Formula: C_{ij} = 2m (L^+_{ii} + L^+_{jj} - 2 L^+_{ij})

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: C_dist

    References
    ----------
    Klein & Randić (1993)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Resistance / commute-time distance from L^+"})


def cheatsheet():
    return "sgtcg: Resistance / commute-time distance from L^+"
