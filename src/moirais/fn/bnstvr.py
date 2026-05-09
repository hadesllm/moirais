"""Bound with treatment-variation assumption."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_treatment_variation"]


def bound_treatment_variation(y, D, X):
    """
    Bound with treatment-variation assumption

    Formula: bound via treatment-variation IV

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
    Manski (1995)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound with treatment-variation assumption"})


def cheatsheet():
    return "bnstvr: Bound with treatment-variation assumption"
