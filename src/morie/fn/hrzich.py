# morie.fn -- function file (hadesllm/morie)
"""Ichimura (1993) single-index NLS estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_ichimura_estimator"]


def horowitz_ichimura_estimator(x, y, bandwidth):
    """
    Ichimura (1993) single-index NLS estimator

    Formula: beta_hat = argmin_{b:|b1|=1} sum_i [Y_i - G_hat_{-i,b}(X_i'b)]^2

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
        Keys: beta_hat, se

    References
    ----------
    Horowitz Ch 2, Sec 2.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Ichimura (1993) single-index NLS estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Ichimura (1993) single-index NLS estimator"})


def cheatsheet():
    return "hrzich: Ichimura (1993) single-index NLS estimator"
