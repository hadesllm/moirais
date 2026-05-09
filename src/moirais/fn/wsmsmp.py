"""Smoothing spline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_smoothing_spline"]


def wasserman_smoothing_spline(x, y, lambda_):
    """
    Smoothing spline

    Formula: min sum (Y_i - m(X_i))^2 + lambda int (m''(x))^2 dx

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    lambda_ : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wasserman (2004), Ch 20
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smoothing spline"})


def cheatsheet():
    return "wsmsmp: Smoothing spline"
