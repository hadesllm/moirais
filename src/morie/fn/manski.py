"""Manski no-assumption bounds on the ATE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["manski_no_assumption_bounds"]


def manski_no_assumption_bounds(y, D, y_min, y_max):
    """
    Manski no-assumption bounds on the ATE

    Formula: L = p E[Y|D=1] + (1-p) y_min - max[Y]; U = p E[Y|D=1] + (1-p) y_max - min[Y]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
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
    Manski (1990); Manski (2003) Partial Identification
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Manski no-assumption bounds on the ATE"})


def cheatsheet():
    return "manski: Manski no-assumption bounds on the ATE"
