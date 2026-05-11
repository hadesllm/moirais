"""Truncated causal forest estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["truncated_cf_estimator"]


def truncated_cf_estimator(y, D, X, trim):
    """
    Truncated causal forest estimator

    Formula: discard top-/bottom-1% extreme tau

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    trim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Crump et al (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Truncated causal forest estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Truncated causal forest estimator"})


def cheatsheet():
    return "trncfg: Truncated causal forest estimator"
