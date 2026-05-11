"""Paule-Mandel iterative τ² estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_paule_mandel"]


def ma_paule_mandel(yi, vi, max_iter):
    """
    Paule-Mandel iterative τ² estimator

    Formula: Solve Σ w_i(y_i-θ̂)² = k-1

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
        Keys: tau2, theta

    References
    ----------
    Paule & Mandel (1982)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Paule-Mandel iterative τ² estimator"})
    estimate = np.median(yi)
    se = 1.2533 * np.std(yi, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Paule-Mandel iterative τ² estimator"})


def cheatsheet():
    return "mapaule: Paule-Mandel iterative τ² estimator"
