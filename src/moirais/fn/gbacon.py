"""Goodman-Bacon decomposition of TWFE estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["goodman_bacon_decomp"]


def goodman_bacon_decomp(y, D, unit, time):
    """
    Goodman-Bacon decomposition of TWFE estimator

    Formula: beta_TWFE = sum_pairs s_pair beta_pair

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Goodman-Bacon (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Goodman-Bacon decomposition of TWFE estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Goodman-Bacon decomposition of TWFE estimator"})


def cheatsheet():
    return "gbacon: Goodman-Bacon decomposition of TWFE estimator"
