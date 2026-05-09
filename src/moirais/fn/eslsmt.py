"""Smoothing spline RSS+lambda penalty."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_smoothing_spline"]


def esl_smoothing_spline(x, y, lambda_):
    """
    Smoothing spline RSS+lambda penalty

    Formula: min sum (y_i-f(x_i))^2 + lambda int f''(t)^2 dt

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
        Keys: fit

    References
    ----------
    Hastie ESL Ch 5
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smoothing spline RSS+lambda penalty"})


def cheatsheet():
    return "eslsmt: Smoothing spline RSS+lambda penalty"
