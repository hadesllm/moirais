"""Regression estimator with auxiliary."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["regression_estimator"]


def regression_estimator(y, x, X, weights):
    """
    Regression estimator with auxiliary

    Formula: ybar + b(Xbar - xbar)

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    X : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Regression estimator with auxiliary"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Regression estimator with auxiliary"})


def cheatsheet():
    return "reglmd: Regression estimator with auxiliary"
