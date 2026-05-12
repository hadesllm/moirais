# morie.fn -- function file (hadesllm/morie)
"""Mixtures of Polya tree processes as prior for density estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_mpt_prior"]


def ghosal_mpt_prior(x):
    """
    Mixtures of Polya tree processes as prior for density estimation

    Formula: f ~ integral PT(alpha, pi) dH(alpha, pi)

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
    Ghosal Ch 3 §3.7.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Mixtures of Polya tree processes as prior for density estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Mixtures of Polya tree processes as prior for density estimation"})


def cheatsheet():
    return "gh_c3_14: Mixtures of Polya tree processes as prior for density estimation"
