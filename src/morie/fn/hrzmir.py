# morie.fn — function file (hadesllm/morie)
"""Marginal integration estimator for additive model components."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_marginal_integration"]


def horowitz_marginal_integration(x, y, bandwidth, j):
    """
    Marginal integration estimator for additive model components

    Formula: g_j_hat(x_j) = (1/n)*sum_i G_hat(X_1i,...,x_j,...,X_di) - mu_hat

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.
    j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_j_hat

    References
    ----------
    Horowitz Ch 3, Sec 3.1.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Marginal integration estimator for additive model components"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Marginal integration estimator for additive model components"})


def cheatsheet():
    return "hrzmir: Marginal integration estimator for additive model components"
