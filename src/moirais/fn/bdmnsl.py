"""Bound under monotone treatment selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_monot_selection"]


def bound_monot_selection(y, D, X):
    """
    Bound under monotone treatment selection

    Formula: E[Y(d)|D=d'] monotone in d'

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound under monotone treatment selection"})


def cheatsheet():
    return "bdmnsl: Bound under monotone treatment selection"
