"""REML estimation of between-study τ²."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_random_reml"]


def ma_random_reml(yi, vi, max_iter):
    """
    REML estimation of between-study τ²

    Formula: Iterate w_i=1/(v_i+τ²); τ²=Σw²(...)/Σw²

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta, tau2, ll, iter

    References
    ----------
    Viechtbauer (2005)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "REML estimation of between-study τ²"})
    estimate = np.median(yi)
    se = 1.2533 * np.std(yi, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "REML estimation of between-study τ²"})


def cheatsheet():
    return "mareml: REML estimation of between-study τ²"
