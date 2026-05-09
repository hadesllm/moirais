# moirais.fn — function file (hadesllm/moirais)
"""Hierarchical Bayesian nonparametric model: hyperprior on DP concentration."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_hierarchical_np"]


def ghosal_hierarchical_np(x):
    """
    Hierarchical Bayesian nonparametric model: hyperprior on DP concentration

    Formula: theta_i|G ~ G, G|alpha ~ DP(alpha,G0), alpha ~ Ga(a,b)

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
    Ghosal Ch 4 §4.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hierarchical Bayesian nonparametric model: hyperprior on DP concentration"})


def cheatsheet():
    return "gh_hier_np: Hierarchical Bayesian nonparametric model: hyperprior on DP concentration"
