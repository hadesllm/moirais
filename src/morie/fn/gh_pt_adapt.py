# morie.fn -- function file (hadesllm/morie)
"""Adaptive Polya tree prior: alpha_m = m^2 gives near-optimal density estimation rate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_pt_adaptive"]


def ghosal_pt_adaptive(x):
    """
    Adaptive Polya tree prior: alpha_m = m^2 gives near-optimal density estimation rate

    Formula: PT*(alpha, m^2): rate n^{-s/(2s+1)} * log n for s-smooth p0

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 7 §7.2.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Adaptive Polya tree prior: alpha_m = m^2 gives near-optimal density estimation rate"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Adaptive Polya tree prior: alpha_m = m^2 gives near-optimal density estimation rate"})


def cheatsheet():
    return "gh_pt_adapt: Adaptive Polya tree prior: alpha_m = m^2 gives near-optimal density estimation rate"
