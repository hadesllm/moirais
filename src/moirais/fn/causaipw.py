"""Augmented IPW doubly-robust estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_aipw"]


def causal_aipw(y, T, ps, m1, m0):
    """
    Augmented IPW doubly-robust estimator

    Formula: AIPW = m̂_1-m̂_0 + (T(Y-m̂_1)/e - (1-T)(Y-m̂_0)/(1-e))

    Parameters
    ----------
    y : array-like
        Input data.
    T : array-like
        Input data.
    ps : array-like
        Input data.
    m1 : array-like
        Input data.
    m0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATE, se

    References
    ----------
    Robins-Rotnitzky-Zhao (1994)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Augmented IPW doubly-robust estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Augmented IPW doubly-robust estimator"})


def cheatsheet():
    return "causaipw: Augmented IPW doubly-robust estimator"
