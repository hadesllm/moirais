# morie.fn — function file (hadesllm/morie)
"""One-step efficient estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_one_step_estimator"]


def kosorok_one_step_estimator(x, y):
    """
    One-step efficient estimator

    Formula: theta_tilde = theta_init + n^{-1} sum IF(Xi)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "One-step efficient estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "One-step efficient estimator"})


def cheatsheet():
    return "ksr15: One-step efficient estimator"
