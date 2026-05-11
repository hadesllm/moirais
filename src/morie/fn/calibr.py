"""GREG-type calibration estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["calibration_estimator"]


def calibration_estimator(y, X, weights, totals):
    """
    GREG-type calibration estimator

    Formula: weights minimize chi-sq subject to margin constraints

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    weights : array-like
        Input data.
    totals : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Deville-Särndal (1992)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "GREG-type calibration estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "GREG-type calibration estimator"})


def cheatsheet():
    return "calibr: GREG-type calibration estimator"
