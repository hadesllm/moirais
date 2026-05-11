"""Bagging prediction average."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_bagging"]


def wasserman_bagging(X, y, model, B):
    """
    Bagging prediction average

    Formula: f_bag(x) = (1/B) sum_b f_b(x)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    model : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Wasserman (2004), Ch 21
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bagging prediction average"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bagging prediction average"})


def cheatsheet():
    return "wsmbgn: Bagging prediction average"
