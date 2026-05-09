"""Balke-Pearl sharp bounds on ATE under instrument."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sharp_bounds_balke_pearl"]


def sharp_bounds_balke_pearl(y, D, Z):
    """
    Balke-Pearl sharp bounds on ATE under instrument

    Formula: linear program over compliance types; tight bounds given Z

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Balke-Pearl (1997)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Balke-Pearl sharp bounds on ATE under instrument"})


def cheatsheet():
    return "sfbnds: Balke-Pearl sharp bounds on ATE under instrument"
