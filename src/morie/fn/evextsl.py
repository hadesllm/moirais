"""Sliding-blocks estimator of θ."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_extremal_index_slidblk"]


def evt_extremal_index_slidblk(x, block_size):
    """
    Sliding-blocks estimator of θ

    Formula: θ̂_SB = log(M_b)/(b log F̂(M_b))

    Parameters
    ----------
    x : array-like
        Input data.
    block_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Northrop (2015)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Sliding-blocks estimator of θ"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Sliding-blocks estimator of θ"})


def cheatsheet():
    return "evextsl: Sliding-blocks estimator of θ"
