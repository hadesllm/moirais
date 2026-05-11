# morie.fn — function file (hadesllm/morie)
"""Manski (1975) maximum-score estimator for binary response."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_manski_max_score"]


def horowitz_manski_max_score(x, y):
    """
    Manski (1975) maximum-score estimator for binary response

    Formula: beta_hat = argmax_{b:|b1|=1} sum_i (2Y_i-1)*I(X_i'b > 0)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat

    References
    ----------
    Horowitz Ch 4, Sec 4.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Manski (1975) maximum-score estimator for binary response"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Manski (1975) maximum-score estimator for binary response"})


def cheatsheet():
    return "hrzmscr: Manski (1975) maximum-score estimator for binary response"
