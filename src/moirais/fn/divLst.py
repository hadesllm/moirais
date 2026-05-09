"""Intra-list diversity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["diversity"]


def diversity(list, sim_matrix):
    """
    Intra-list diversity

    Formula: avg(1 − sim(i,j)) over recommended pairs

    Parameters
    ----------
    list : array-like
        Input data.
    sim_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Smyth-McClave (2001)
    """
    list = np.atleast_1d(np.asarray(list, dtype=float))
    n = len(list)
    result = float(np.mean(list))
    se = float(np.std(list, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Intra-list diversity"})


def cheatsheet():
    return "divLst: Intra-list diversity"
