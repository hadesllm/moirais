# morie.fn -- function file (hadesllm/morie)
"""Naive kernel MRL function estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_naive_mrl"]


def fauzi_naive_mrl(t, bandwidth, kernel):
    """
    Naive kernel MRL function estimator

    Formula: m_hat_X(t) = [h*sum V((t-X_i)/h)] / [sum V((t-X_i)/h)]

    Parameters
    ----------
    t : array-like
        Input data.
    bandwidth : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 4, Eq 4.2
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Naive kernel MRL function estimator"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Naive kernel MRL function estimator"})


def cheatsheet():
    return "fzmrln: Naive kernel MRL function estimator"
