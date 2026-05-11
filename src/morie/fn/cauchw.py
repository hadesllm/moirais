"""Cauchy weight function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cauchy_weight"]


def cauchy_weight(y, c):
    """
    Cauchy weight function

    Formula: w(r) = 1/(1 + (r/c)^2)

    Parameters
    ----------
    y : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Holland & Welsch (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cauchy weight function"})


def cheatsheet():
    return "cauchw: Cauchy weight function"
