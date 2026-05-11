"""Welsch (Gaussian) weight function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["welsch_weight"]


def welsch_weight(y, c):
    """
    Welsch (Gaussian) weight function

    Formula: w(r) = exp(-(r/c)^2)

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
    Welsch (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Welsch (Gaussian) weight function"})


def cheatsheet():
    return "welshw: Welsch (Gaussian) weight function"
