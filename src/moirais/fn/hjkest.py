"""Hájek ratio-of-weights estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hajek_estimator"]


def hajek_estimator(y, pi):
    """
    Hájek ratio-of-weights estimator

    Formula: (sum y_i / pi_i) / (sum 1 / pi_i)

    Parameters
    ----------
    y : array-like
        Input data.
    pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hájek (1971)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Hájek ratio-of-weights estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Hájek ratio-of-weights estimator"})


def cheatsheet():
    return "hjkest: Hájek ratio-of-weights estimator"
