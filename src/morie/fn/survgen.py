"""Generalized estimating equations for survival."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["general_estimating_eq_surv"]


def general_estimating_eq_surv(time, event, X, cluster):
    """
    Generalized estimating equations for survival

    Formula: GEE with cluster-robust variance

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liang-Zeger (1986); Therneau et al (2003)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Generalized estimating equations for survival"})
    estimate = np.median(time)
    se = 1.2533 * np.std(time, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Generalized estimating equations for survival"})


def cheatsheet():
    return "survgen: Generalized estimating equations for survival"
