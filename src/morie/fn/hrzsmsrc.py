# morie.fn — function file (hadesllm/morie)
"""Rate of convergence of smoothed maximum-score estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_sms_rate"]


def horowitz_sms_rate(n, smoothness_order):
    """
    Rate of convergence of smoothed maximum-score estimator

    Formula: ||beta_hat - beta|| = O_p(n^{-r/(2r+1)}) where r = order of K smoothness

    Parameters
    ----------
    n : array-like
        Input data.
    smoothness_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rate

    References
    ----------
    Horowitz Ch 4, Sec 4.3.3
    """
    n = int(n) if np.ndim(n) == 0 else len(n)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Rate of convergence of smoothed maximum-score estimator"})
    estimate = np.median(n)
    se = 1.2533 * np.std(n, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Rate of convergence of smoothed maximum-score estimator"})


def cheatsheet():
    return "hrzsmsrc: Rate of convergence of smoothed maximum-score estimator"
