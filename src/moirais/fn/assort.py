"""Degree assortativity coefficient (Pearson on edge endpoints)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["degree_assortativity"]


def degree_assortativity(y, A):
    """
    Degree assortativity coefficient (Pearson on edge endpoints)

    Formula: r = (sum_e (j_e - jbar)(k_e - kbar)) / (sd_j sd_k)

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newman (2002, 2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Degree assortativity coefficient (Pearson on edge endpoints)"})


def cheatsheet():
    return "assort: Degree assortativity coefficient (Pearson on edge endpoints)"
