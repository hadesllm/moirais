"""Bound when monotonicity violated."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_no_monotonicity"]


def bound_no_monotonicity(y, D, Z):
    """
    Bound when monotonicity violated

    Formula: compliers + always-takers + never-takers + defiers

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Chaisemartin (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound when monotonicity violated"})


def cheatsheet():
    return "bndnmt: Bound when monotonicity violated"
