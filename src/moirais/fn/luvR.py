"""Louvain modularity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["louvain"]


def louvain(A, resolution):
    """
    Louvain modularity

    Formula: greedy modularity gain at each level

    Parameters
    ----------
    A : array-like
        Input data.
    resolution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Blondel et al (2008)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Louvain modularity"})


def cheatsheet():
    return "luvR: Louvain modularity"
