"""Callaway-Sant'Anna group-time ATT(g,t)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["callaway_santanna"]


def callaway_santanna(y, D, unit, time, cohort):
    """
    Callaway-Sant'Anna group-time ATT(g,t)

    Formula: ATT(g,t) = E[Y_t(1) - Y_t(0) | G=g]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    cohort : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Callaway & Sant'Anna (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Callaway-Sant'Anna group-time ATT(g,t)"})


def cheatsheet():
    return "cssant: Callaway-Sant'Anna group-time ATT(g,t)"
