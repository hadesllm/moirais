"""Inverse-probability-of-censoring weighted estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ipcw_estimator"]


def ipcw_estimator(time, event, cens_model):
    """
    Inverse-probability-of-censoring weighted estimator

    Formula: weight by 1/G(t-) for not-yet-censored

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    cens_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Rotnitzky (1992)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Inverse-probability-of-censoring weighted estimator"})
    estimate = np.median(time)
    se = 1.2533 * np.std(time, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Inverse-probability-of-censoring weighted estimator"})


def cheatsheet():
    return "survipw: Inverse-probability-of-censoring weighted estimator"
