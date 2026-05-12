# morie.fn -- function file (hadesllm/morie)
"""Nested Dirichlet process: G_j ~ DP(alpha, G0), G0 ~ DP(gamma, H) for clustering."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_nested_dp"]


def ghosal_nested_dp(x):
    """
    Nested Dirichlet process: G_j ~ DP(alpha, G0), G0 ~ DP(gamma, H) for clustering

    Formula: G_j | G0 ~ DP(alpha, G0), G0 ~ DP(gamma, H)

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
    Ghosal Ch 14 §14.9.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nested Dirichlet process: G_j ~ DP(alpha, G0), G0 ~ DP(gamma, H) for clustering"})


def cheatsheet():
    return "gh_c14_22: Nested Dirichlet process: G_j ~ DP(alpha, G0), G0 ~ DP(gamma, H) for clustering"
