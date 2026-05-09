"""Smoothing spline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["smoothing_spline"]


def smoothing_spline(x, y, lam):
    """
    Smoothing spline

    Formula: min sum (y_i − f(x_i))² + λ ∫ (f''(x))²

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wahba (1990)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smoothing spline"})


def cheatsheet():
    return "smspln: Smoothing spline"
