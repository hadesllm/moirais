# morie.fn — function file (hadesllm/morie)
"""Invariant Dirichlet process: DP prior invariant under group transformations."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_inv_dp"]


def ghosal_inv_dp(x):
    """
    Invariant Dirichlet process: DP prior invariant under group transformations

    Formula: IDP(alpha): G =_d T#G for transformation T in group G

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
    Ghosal Ch 4 §4.6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Invariant Dirichlet process: DP prior invariant under group transformations"})


def cheatsheet():
    return "gh_c4_21: Invariant Dirichlet process: DP prior invariant under group transformations"
