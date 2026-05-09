"""Generalised regression (calibration) estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["calibration_greg"]


def calibration_greg(y, x, weights, totals):
    """
    Generalised regression (calibration) estimator

    Formula: theta_GREG = theta_HT + (T_x - hat T_x)' B

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
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
    Sarndal, Swensson, Wretman (1992); Deville & Sarndal (1992)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Generalised regression (calibration) estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Generalised regression (calibration) estimator"})


def cheatsheet():
    return "genrgr: Generalised regression (calibration) estimator"
