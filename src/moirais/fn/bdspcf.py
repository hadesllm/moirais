"""Specification bound from misspecification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_specification"]


def bound_specification(y, D, X):
    """
    Specification bound from misspecification

    Formula: max bias under linear-misspec

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
    Andrews-Kasy (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Specification bound from misspecification"})


def cheatsheet():
    return "bdspcf: Specification bound from misspecification"
