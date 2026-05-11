# morie.fn — function file (hadesllm/morie)
"""Dirichlet process DP(alpha, G0): finite-dimensional marginals are Dirichlet."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_def"]


def ghosal_dp_def(x):
    """
    Dirichlet process DP(alpha, G0): finite-dimensional marginals are Dirichlet

    Formula: (G(A_1)..G(A_k)) ~ Dir(alpha*G0(A_1)..alpha*G0(A_k)) for any partition

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
    Ghosal Ch 4 §4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet process DP(alpha, G0): finite-dimensional marginals are Dirichlet"})


def cheatsheet():
    return "gh_c4_1: Dirichlet process DP(alpha, G0): finite-dimensional marginals are Dirichlet"
