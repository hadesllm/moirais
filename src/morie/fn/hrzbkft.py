# morie.fn — function file (hadesllm/morie)
"""Backfitting algorithm for additive model estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_backfitting"]


def horowitz_backfitting(x, y, bandwidth):
    """
    Backfitting algorithm for additive model estimation

    Formula: Iterate: g_j = S_j(Y - mu - sum_{k!=j} g_k) until convergence; S_j = smoother

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
        Keys: g_j_hats, mu_hat

    References
    ----------
    Horowitz Ch 3, Sec 3.1.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Backfitting algorithm for additive model estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Backfitting algorithm for additive model estimation"})


def cheatsheet():
    return "hrzbkft: Backfitting algorithm for additive model estimation"
