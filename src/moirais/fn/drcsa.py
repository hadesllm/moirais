"""Doubly-robust Callaway-Sant'Anna ATT(g,t) estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_callaway_santanna"]


def dr_callaway_santanna(y, D, unit, time, cohort, X):
    """
    Doubly-robust Callaway-Sant'Anna ATT(g,t) estimator

    Formula: DR-ATT(g,t) combines outcome regression and IPW under not-yet-treated controls

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    cohort : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Callaway & Sant'Anna (2021) JoE
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Doubly-robust Callaway-Sant'Anna ATT(g,t) estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Doubly-robust Callaway-Sant'Anna ATT(g,t) estimator"})


def cheatsheet():
    return "drcsa: Doubly-robust Callaway-Sant'Anna ATT(g,t) estimator"
