"""Degree diagonal matrix from adjacency."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_degree_matrix"]


def sgt_degree_matrix(A):
    """
    Degree diagonal matrix from adjacency

    Formula: D_{ii} = Σ_j A_{ij}

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: D

    References
    ----------
    Chung (1997)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Degree diagonal matrix from adjacency"})


def cheatsheet():
    return "sgtdeg: Degree diagonal matrix from adjacency"
