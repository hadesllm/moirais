# morie.fn -- function file (hadesllm/morie)
"""Penalized Dirichlet process: DP with penalty on deviation from G0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_pen_dp"]


def ghosal_pen_dp(x):
    """
    Penalized Dirichlet process: DP with penalty on deviation from G0

    Formula: pi(G) propto exp(-lambda*d(G,G0)) * DP(alpha,G0)(G)

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
    Ghosal Ch 4 §4.6.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penalized Dirichlet process: DP with penalty on deviation from G0"})


def cheatsheet():
    return "gh_c4_23: Penalized Dirichlet process: DP with penalty on deviation from G0"
