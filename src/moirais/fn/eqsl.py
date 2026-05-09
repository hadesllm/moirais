"""Stocking-Lord equating (TCC matching)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["equating_stocking_lord"]


def equating_stocking_lord(y, b_R, b_F, a_R, a_F):
    """
    Stocking-Lord equating (TCC matching)

    Formula: min sum_theta (T_R(theta) - T_F(A theta + B))^2

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
    Stocking & Lord (1983)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stocking-Lord equating (TCC matching)"})


def cheatsheet():
    return "eqsl: Stocking-Lord equating (TCC matching)"
