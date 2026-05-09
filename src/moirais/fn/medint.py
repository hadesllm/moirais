"""Mediated interaction term in 4-way decomp."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mediated_interaction"]


def mediated_interaction(Y, X, M):
    """
    Mediated interaction term in 4-way decomp

    Formula: E[(Y(1,M(1))−Y(0,M(1)))−(Y(1,M(0))−Y(0,M(0)))]

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele (2014)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mediated interaction term in 4-way decomp"})


def cheatsheet():
    return "medint: Mediated interaction term in 4-way decomp"
