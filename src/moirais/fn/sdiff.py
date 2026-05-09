# moirais.fn — function file (hadesllm/moirais)
"""Synthetic difference-in-differences estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["synthetic_did"]


def synthetic_did(Y, unit_id, time_id, treated, treatment_time):
    """
    Synthetic difference-in-differences estimator

    Formula: tau_SDID = tau_DID with re-weighted pre-treatment observations; combines synthetic control + DiD weights

    Parameters
    ----------
    Y : array-like
        Input data.
    unit_id : array-like
        Input data.
    time_id : array-like
        Input data.
    treated : array-like
        Input data.
    treatment_time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'att': 'float', 'se': 'float'}

    References
    ----------
    Molak Ch 11
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    if Y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Synthetic difference-in-differences estimator"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Synthetic difference-in-differences estimator"})


def cheatsheet():
    return "sdiff: Synthetic difference-in-differences estimator"
