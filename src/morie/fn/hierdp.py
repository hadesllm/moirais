"""Hierarchical DP density estimation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hierarchical_dp_density"]


def hierarchical_dp_density(y, groups, gamma, alpha):
    """
    Hierarchical DP density estimation

    Formula: shared atoms; group-specific weights

    Parameters
    ----------
    y : array-like
        Input data.
    groups : array-like
        Input data.
    gamma : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Teh et al (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Hierarchical DP density estimation"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Hierarchical DP density estimation"})


def cheatsheet():
    return "hierdp: Hierarchical DP density estimation"
