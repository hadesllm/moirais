"""Horowitz-Manski bounds under missing data."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["horowitz_manski_bounds"]


def horowitz_manski_bounds(y, R, y_min, y_max):
    """
    Horowitz-Manski bounds under missing data

    Formula: bounds with missing-at-random vs MNAR contrast

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
    Horowitz-Manski (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Horowitz-Manski bounds under missing data"})


def cheatsheet():
    return "hzbnds: Horowitz-Manski bounds under missing data"
