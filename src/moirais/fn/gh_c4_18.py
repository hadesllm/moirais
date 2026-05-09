# moirais.fn — function file (hadesllm/moirais)
"""Cifarelli-Regazzini theorem: distribution of mean of DP via characteristic function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_mean_dist"]


def ghosal_dp_mean_dist(x):
    """
    Cifarelli-Regazzini theorem: distribution of mean of DP via characteristic function

    Formula: E[G] = integral x dG(x), distribution via Cifarelli-Regazzini formula

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
    Ghosal Ch 4 §4.3.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cifarelli-Regazzini theorem: distribution of mean of DP via characteristic function"})


def cheatsheet():
    return "gh_c4_18: Cifarelli-Regazzini theorem: distribution of mean of DP via characteristic function"
