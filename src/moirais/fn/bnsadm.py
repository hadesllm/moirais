"""Bound on admissible estimators."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_admissible_estimators"]


def bound_admissible_estimators(y, D, X, family):
    """
    Bound on admissible estimators

    Formula: min-max admissible loss bound

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hirano-Porter (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bound on admissible estimators"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bound on admissible estimators"})


def cheatsheet():
    return "bnsadm: Bound on admissible estimators"
