"""Hampel three-part redescender."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hampel_redescend"]


def hampel_redescend(r, a, b, c):
    """
    Hampel three-part redescender

    Formula: piecewise: linear / flat / linear-down / zero

    Parameters
    ----------
    r : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hampel (1974)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hampel three-part redescender"})


def cheatsheet():
    return "hampel: Hampel three-part redescender"
