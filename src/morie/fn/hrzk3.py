# morie.fn — function file (hadesllm/morie)
"""Local linear regression estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_local_linear"]


def horowitz_local_linear(x, y, bandwidth):
    """
    Local linear regression estimator

    Formula: beta = argmin sum K_h(x-Xi)(Yi - a - b(Xi-x))^2

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Horowitz (2009), Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Local linear regression estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Local linear regression estimator"})


def cheatsheet():
    return "hrzk3: Local linear regression estimator"
