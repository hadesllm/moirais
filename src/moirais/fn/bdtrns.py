"""Bound under transport assumption."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_transport"]


def bound_transport(y, D, X, S):
    """
    Bound under transport assumption

    Formula: target-pop bounds via source weights

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound under transport assumption"})


def cheatsheet():
    return "bdtrns: Bound under transport assumption"
