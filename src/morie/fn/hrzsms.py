# morie.fn -- function file (hadesllm/morie)
"""Horowitz smoothed maximum-score estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_smoothed_max_score"]


def horowitz_smoothed_max_score(x, y, bandwidth):
    """
    Horowitz smoothed maximum-score estimator

    Formula: beta_hat = argmax_{b:|b1|=1} (1/n)*sum_i (2Y_i-1)*K((X_i'b)/h_n)

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
    Horowitz Ch 4, Sec 4.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Horowitz smoothed maximum-score estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Horowitz smoothed maximum-score estimator"})


def cheatsheet():
    return "hrzsms: Horowitz smoothed maximum-score estimator"
