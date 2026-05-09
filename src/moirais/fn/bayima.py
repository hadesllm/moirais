"""Importance sampling estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["importance_sampling"]


def importance_sampling(target, proposal, n):
    """
    Importance sampling estimator

    Formula: E[f] ≈ sum w_i f(x_i) / sum w_i

    Parameters
    ----------
    target : array-like
        Input data.
    proposal : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geweke (1989)
    """
    target = np.atleast_1d(np.asarray(target, dtype=float))
    n = len(target)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Importance sampling estimator"})
    estimate = np.median(target)
    se = 1.2533 * np.std(target, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Importance sampling estimator"})


def cheatsheet():
    return "bayima: Importance sampling estimator"
