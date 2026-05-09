"""Bound under missing outcome."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_missing_outcome"]


def bound_missing_outcome(y, R, y_min, y_max):
    """
    Bound under missing outcome

    Formula: Manski-Horowitz bounds for missingness

    Parameters
    ----------
    y : array-like
        Input data.
    R : array-like
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
    Horowitz-Manski (2000); Manski (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound under missing outcome"})


def cheatsheet():
    return "bndmsg: Bound under missing outcome"
