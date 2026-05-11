"""Farey sequence enumeration."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["farey_seq"]


def farey_seq(n):
    """
    Farey sequence enumeration

    Formula: all reduced fractions p/q with q≤n

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical number theory
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Farey sequence enumeration"})


def cheatsheet():
    return "diopT: Farey sequence enumeration"
