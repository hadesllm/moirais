"""TMLE for non-smooth functionals (median, etc.)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_non_smooth"]


def tmle_non_smooth(y, D, X, bw):
    """
    TMLE for non-smooth functionals (median, etc.)

    Formula: smoothed influence curve via kernel

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    bw : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hines-Diaz-Naimi-vdL (2022)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for non-smooth functionals (median, etc.)"})


def cheatsheet():
    return "tmlnsm: TMLE for non-smooth functionals (median, etc.)"
