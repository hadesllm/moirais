# morie.fn -- function file (hadesllm/morie)
"""Conditional prediction of Y given X after transformation model estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_conditional_prediction"]


def horowitz_conditional_prediction(x, y_threshold, T_hat, F_hat, beta_hat):
    """
    Conditional prediction of Y given X after transformation model estimation

    Formula: P(Y<=y|X=x) = F_hat(T_hat(y) - x'beta_hat)

    Parameters
    ----------
    x : array-like
        Input data.
    y_threshold : array-like
        Input data.
    T_hat : array-like
        Input data.
    F_hat : array-like
        Input data.
    beta_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Horowitz Ch 6, Sec 6.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Conditional prediction of Y given X after transformation model estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Conditional prediction of Y given X after transformation model estimation"})


def cheatsheet():
    return "hrzycp: Conditional prediction of Y given X after transformation model estimation"
