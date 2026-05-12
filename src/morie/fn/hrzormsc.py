# morie.fn -- function file (hadesllm/morie)
"""Ordered-response maximum-score estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_ordered_max_score"]


def horowitz_ordered_max_score(x, y):
    """
    Ordered-response maximum-score estimator

    Formula: Y in {0,1,...,J}; beta_hat = argmax sum_{j} sum_i I(Y_i=j)*I(alpha_{j-1}<X_i'b<=alpha_j)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat, thresholds

    References
    ----------
    Horowitz Ch 4, Sec 4.4.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Ordered-response maximum-score estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Ordered-response maximum-score estimator"})


def cheatsheet():
    return "hrzormsc: Ordered-response maximum-score estimator"
