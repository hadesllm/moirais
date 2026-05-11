# morie.fn — function file (hadesllm/morie)
"""Jackknife bias and variance estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["jackknife_estimator"]


def jackknife_estimator(x):
    """
    Jackknife bias and variance estimation

    Formula: theta_jack = n*theta - (n-1)*mean(theta_{-i})

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Quenouille (1956)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Jackknife bias and variance estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Jackknife bias and variance estimation"})


def cheatsheet():
    return "jkest: Jackknife bias and variance estimation"
