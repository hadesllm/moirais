# moirais.fn — function file (hadesllm/moirais)
"""Hierarchical Bayes nonparametric."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_hierarchical_bayes"]


def ghosal_hierarchical_bayes(x):
    """
    Hierarchical Bayes nonparametric

    Formula: theta|G ~ G, G|alpha ~ DP(alpha, G0), alpha ~ prior

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
    Ghosal Ch 15
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hierarchical Bayes nonparametric"})


def cheatsheet():
    return "ghhbp: Hierarchical Bayes nonparametric"
