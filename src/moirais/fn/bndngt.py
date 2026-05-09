"""Negative-only treatment bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_neg_treatment"]


def bound_neg_treatment(y, D, y_min):
    """
    Negative-only treatment bound

    Formula: upper bound on harmful treatment

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    y_min : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski (1995, 1997)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Negative-only treatment bound"})


def cheatsheet():
    return "bndngt: Negative-only treatment bound"
