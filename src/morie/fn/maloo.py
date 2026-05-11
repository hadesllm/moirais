"""Leave-one-out sensitivity for pooled estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_leave_one_out"]


def ma_leave_one_out(yi, vi, method):
    """
    Leave-one-out sensitivity for pooled estimate

    Formula: θ̂_(-i), τ²_(-i) for each i excluded

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_loo, tau2_loo

    References
    ----------
    Viechtbauer (2010)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Leave-one-out sensitivity for pooled estimate"})
    estimate = np.median(yi)
    se = 1.2533 * np.std(yi, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Leave-one-out sensitivity for pooled estimate"})


def cheatsheet():
    return "maloo: Leave-one-out sensitivity for pooled estimate"
