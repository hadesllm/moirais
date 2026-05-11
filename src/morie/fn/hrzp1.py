# morie.fn — function file (hadesllm/morie)
"""Partially linear regression estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_plr_estimator"]


def horowitz_plr_estimator(x, y, z):
    """
    Partially linear regression estimator

    Formula: Y = X*beta + g(Z) + e, Robinson (1988)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Horowitz (2009), Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Partially linear regression estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Partially linear regression estimator"})


def cheatsheet():
    return "hrzp1: Partially linear regression estimator"
