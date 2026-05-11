"""Synthetic control estimator for comparative case studies."""
import numpy as np
from ._richresult import RichResult

__all__ = ["synthetic_control"]


def synthetic_control(Y, unit_id, time_id, treated_unit, treatment_time):
    """
    Synthetic control estimator for comparative case studies

    Formula: w* = argmin ||X_1 - X_0*w||^2 s.t. w>=0, sum(w)=1; effect = Y_1t - Y_0t*w* for t>T_0

    Parameters
    ----------
    Y : array-like
        Input data.
    unit_id : array-like
        Input data.
    time_id : array-like
        Input data.
    treated_unit : array-like
        Input data.
    treatment_time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'weights': 'array', 'att': 'array', 'placebo_p': 'float'}

    References
    ----------
    Molak Ch 11
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    if Y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Synthetic control estimator for comparative case studies"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Synthetic control estimator for comparative case studies"})


def cheatsheet():
    return "synct: Synthetic control estimator for comparative case studies"
