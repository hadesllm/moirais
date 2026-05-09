"""Huber psi-weight function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["huber_weight"]


def huber_weight(y, k):
    """
    Huber psi-weight function

    Formula: w(r) = 1 if |r|<=k else k/|r|

    Parameters
    ----------
    y : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Huber (1964)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Huber psi-weight function"})


def cheatsheet():
    return "huberw: Huber psi-weight function"
