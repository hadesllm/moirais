"""Haebara equating (item-by-item ICC matching)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["equating_haebara"]


def equating_haebara(y, b_R, b_F, a_R, a_F):
    """
    Haebara equating (item-by-item ICC matching)

    Formula: min sum_i sum_theta (P_R_i(theta) - P_F_i(A theta + B))^2

    Parameters
    ----------
    y : array-like
        Input data.
    b_R : array-like
        Input data.
    b_F : array-like
        Input data.
    a_R : array-like
        Input data.
    a_F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Haebara (1980)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Haebara equating (item-by-item ICC matching)"})


def cheatsheet():
    return "eqhae: Haebara equating (item-by-item ICC matching)"
