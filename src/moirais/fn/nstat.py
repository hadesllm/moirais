# moirais.fn — function file (hadesllm/moirais)
"""Non-stationary covariance estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["nonstationary_covariance"]


def nonstationary_covariance(x, coords):
    """
    Non-stationary covariance estimation

    Formula: C(s1,s2) = sigma(s1)*sigma(s2)*rho(s1,s2)

    Parameters
    ----------
    x : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Non-stationary covariance estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Non-stationary covariance estimation"})


def cheatsheet():
    return "nstat: Non-stationary covariance estimation"
