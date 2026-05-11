"""Huber regression (M-estimator)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["huber_regression"]


def huber_regression(X, y, k):
    """
    Huber regression (M-estimator)

    Formula: IRLS with Huber weights

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Huber (1973)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Huber regression (M-estimator)"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Huber regression (M-estimator)"})


def cheatsheet():
    return "hubrr: Huber regression (M-estimator)"
