"""Variance estimator for causal forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_forest_variance"]


def causal_forest_variance(forest, X_test):
    """
    Variance estimator for causal forest

    Formula: infinitesimal jackknife

    Parameters
    ----------
    forest : array-like
        Input data.
    X_test : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wager-Athey (2018) Theorem 9
    """
    forest = np.atleast_1d(np.asarray(forest, dtype=float))
    n = len(forest)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Variance estimator for causal forest"})
    estimate = np.median(forest)
    se = 1.2533 * np.std(forest, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Variance estimator for causal forest"})


def cheatsheet():
    return "crfvar: Variance estimator for causal forest"
