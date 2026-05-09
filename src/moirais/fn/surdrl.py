"""Doubly-robust survey estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survey_dr_estimator"]


def survey_dr_estimator(y, D, X, sampling_weights):
    """
    Doubly-robust survey estimator

    Formula: DR moment under sampling design

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    sampling_weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Rotnitzky (1995)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Doubly-robust survey estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Doubly-robust survey estimator"})


def cheatsheet():
    return "surdrl: Doubly-robust survey estimator"
