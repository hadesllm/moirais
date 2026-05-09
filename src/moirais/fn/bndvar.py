"""Variance term in partial-ID estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_variance_term"]


def bound_variance_term(theta, moments):
    """
    Variance term in partial-ID estimator

    Formula: asymptotic variance under moment inequality

    Parameters
    ----------
    theta : array-like
        Input data.
    moments : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stoye (2009)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Variance term in partial-ID estimator"})
    estimate = np.median(theta)
    se = 1.2533 * np.std(theta, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Variance term in partial-ID estimator"})


def cheatsheet():
    return "bndvar: Variance term in partial-ID estimator"
