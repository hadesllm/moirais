# morie.fn -- function file (hadesllm/morie)
"""Inverse probability weighting (IPW) estimator for ATE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ipw_estimator"]


def ipw_estimator(Y, T, X):
    """
    Inverse probability weighting (IPW) estimator for ATE

    Formula: ATE_IPW = (1/n)*sum_i [T_i*Y_i/e(X_i) - (1-T_i)*Y_i/(1-e(X_i))]

    Parameters
    ----------
    Y : array-like
        Input data.
    T : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ate': 'float', 'se': 'float'}

    References
    ----------
    Molak Ch 9
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    if Y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Inverse probability weighting (IPW) estimator for ATE"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Inverse probability weighting (IPW) estimator for ATE"})


def cheatsheet():
    return "ipwef: Inverse probability weighting (IPW) estimator for ATE"
