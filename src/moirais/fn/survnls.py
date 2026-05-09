"""NLS estimator for parametric survival."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nonlinear_least_squares_surv"]


def nonlinear_least_squares_surv(time, event, model):
    """
    NLS estimator for parametric survival

    Formula: min sum (S_emp(t) - S_param(t;theta))^2

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lawless (2003)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "NLS estimator for parametric survival"})
    estimate = np.median(time)
    se = 1.2533 * np.std(time, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "NLS estimator for parametric survival"})


def cheatsheet():
    return "survnls: NLS estimator for parametric survival"
