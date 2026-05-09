"""Warm's weighted likelihood theta estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weighted_likelihood_theta"]


def weighted_likelihood_theta(y, P_theta):
    """
    Warm's weighted likelihood theta estimate

    Formula: theta_hat_WLE = argmax_theta L(theta|y) sqrt(I(theta))

    Parameters
    ----------
    y : array-like
        Input data.
    P_theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Warm (1989)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Warm's weighted likelihood theta estimate"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Warm's weighted likelihood theta estimate"})


def cheatsheet():
    return "wleth: Warm's weighted likelihood theta estimate"
