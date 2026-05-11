"""Variance estimator for Cox beta."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variance_cox_estimator"]


def variance_cox_estimator(fit):
    """
    Variance estimator for Cox beta

    Formula: inverse of observed information

    Parameters
    ----------
    fit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cox (1975)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Variance estimator for Cox beta"})
    estimate = np.median(fit)
    se = 1.2533 * np.std(fit, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Variance estimator for Cox beta"})


def cheatsheet():
    return "survvar: Variance estimator for Cox beta"
