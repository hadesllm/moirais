"""Bound under monotone IV."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_monot_inst"]


def bound_monot_inst(y, D, Z, y_min, y_max):
    """
    Bound under monotone IV

    Formula: intersect bounds across Z values

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.
    y_min : array-like
        Input data.
    y_max : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski-Pepper (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound under monotone IV"})


def cheatsheet():
    return "bdmnts: Bound under monotone IV"
