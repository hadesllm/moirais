"""Resistance distance matrix from L^+."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_resistance_distance_matrix"]


def sgt_resistance_distance_matrix(A):
    """
    Resistance distance matrix from L^+

    Formula: R_{ij} = L^+_{ii} + L^+_{jj} - 2 L^+_{ij}

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: R

    References
    ----------
    Klein & Randić (1993)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Resistance distance matrix from L^+"})


def cheatsheet():
    return "sgtreg: Resistance distance matrix from L^+"
