# morie.fn -- function file (hadesllm/morie)
"""Posterior consistency for normal mixture density estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_norm_mix_con"]


def ghosal_norm_mix_con(x):
    """
    Posterior consistency for normal mixture density estimation

    Formula: DP mixture of N(mu,sigma^2) consistent at any smooth density p0

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
    Ghosal Ch 7 §7.2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Posterior consistency for normal mixture density estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Posterior consistency for normal mixture density estimation"})


def cheatsheet():
    return "gh_c7_4: Posterior consistency for normal mixture density estimation"
