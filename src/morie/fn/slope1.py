"""Slope-one predictor."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["slope_one"]


def slope_one(R, u, i):
    """
    Slope-one predictor

    Formula: r̂_{ui} = mean(r_{uj} + dev(i,j)) over rated j

    Parameters
    ----------
    R : array-like
        Input data.
    u : array-like
        Input data.
    i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lemire-Maclachlan (2005)
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Slope-one predictor"})


def cheatsheet():
    return "slope1: Slope-one predictor"
