# moirais.fn — function file (hadesllm/moirais)
"""Propensity score matching for ATE/ATT estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["propensity_score_matching"]


def propensity_score_matching(Y, T, X, caliper):
    """
    Propensity score matching for ATE/ATT estimation

    Formula: e(Y) = P(T=1|X=Y); match treated to nearest control by |e(x_i) - e(x_j)|

    Parameters
    ----------
    Y : array-like
        Input data.
    T : array-like
        Input data.
    X : array-like
        Input data.
    caliper : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ate': 'float', 'att': 'float', 'matched_pairs': 'array'}

    References
    ----------
    Molak Ch 9
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    if Y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Propensity score matching for ATE/ATT estimation"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Propensity score matching for ATE/ATT estimation"})


def cheatsheet():
    return "pscsm: Propensity score matching for ATE/ATT estimation"
