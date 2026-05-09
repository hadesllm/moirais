# moirais.fn — function file (hadesllm/moirais)
"""Relationship between W and average Spearman rho."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_concordance_rho_link"]


def gibbons_concordance_rho_link(W, k):
    """
    Relationship between W and average Spearman rho

    Formula: rbar = (k*W - 1)/(k - 1); W and rbar are equivalent summaries

    Parameters
    ----------
    W : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rbar

    References
    ----------
    Gibbons Ch 12.4
    """
    W = np.asarray(W, dtype=float)
    n = int(W) if W.ndim == 0 else len(W)
    if W.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Relationship between W and average Spearman rho"})
    estimate = np.median(W)
    se = 1.2533 * np.std(W, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Relationship between W and average Spearman rho"})


def cheatsheet():
    return "gb1241r: Relationship between W and average Spearman rho"
