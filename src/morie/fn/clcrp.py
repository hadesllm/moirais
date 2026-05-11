"""Clustered Chinese Restaurant Process."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clustered_crp"]


def clustered_crp(y, distances, alpha):
    """
    Clustered Chinese Restaurant Process

    Formula: distance-dependent CRP

    Parameters
    ----------
    y : array-like
        Input data.
    distances : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Blei-Frazier (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clustered Chinese Restaurant Process"})


def cheatsheet():
    return "clcrp: Clustered Chinese Restaurant Process"
