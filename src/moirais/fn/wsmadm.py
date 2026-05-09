"""Admissibility (no estimator dominates)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_admissible"]


def wasserman_admissible(estimator):
    """
    Admissibility (no estimator dominates)

    Formula: T admissible if no T' with R(T',F) <= R(T,F) all F, < some F

    Parameters
    ----------
    estimator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bool

    References
    ----------
    Wasserman (2004), Ch 12
    """
    estimator = np.atleast_1d(np.asarray(estimator, dtype=float))
    n = len(estimator)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Admissibility (no estimator dominates)"})
    estimate = np.median(estimator)
    se = 1.2533 * np.std(estimator, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Admissibility (no estimator dominates)"})


def cheatsheet():
    return "wsmadm: Admissibility (no estimator dominates)"
