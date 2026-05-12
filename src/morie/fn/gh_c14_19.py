# morie.fn -- function file (hadesllm/morie)
"""Local Dirichlet process: DP with location-dependent weights for regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_local_dp"]


def ghosal_local_dp(x):
    """
    Local Dirichlet process: DP with location-dependent weights for regression

    Formula: G(x,.) = sum_k w_k(x) delta_{theta_k}, w_k(x) from kernel stick-breaking

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
    Ghosal Ch 14 §14.9.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local Dirichlet process: DP with location-dependent weights for regression"})


def cheatsheet():
    return "gh_c14_19: Local Dirichlet process: DP with location-dependent weights for regression"
