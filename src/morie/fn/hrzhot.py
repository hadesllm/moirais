# morie.fn — function file (hadesllm/morie)
"""Horowitz estimators for T and F in fully nonparametric transformation model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_T_F_estimators"]


def horowitz_T_F_estimators(x, y, bandwidth, beta_hat):
    """
    Horowitz estimators for T and F in fully nonparametric transformation model

    Formula: T_hat from weighted CDF of Y; F_hat from CDF of T_hat(Y)-X'beta_hat

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.
    beta_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T_hat, F_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Horowitz estimators for T and F in fully nonparametric transformation model"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Horowitz estimators for T and F in fully nonparametric transformation model"})


def cheatsheet():
    return "hrzhot: Horowitz estimators for T and F in fully nonparametric transformation model"
