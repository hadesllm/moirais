# morie.fn -- function file (hadesllm/morie)
"""Finite-dimensional Dirichlet distribution: Dir(alpha_1..alpha_k) on simplex S_k."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_fin_dir_def"]


def ghosal_fin_dir_def(x):
    """
    Finite-dimensional Dirichlet distribution: Dir(alpha_1..alpha_k) on simplex S_k

    Formula: p(x_1..x_k) propto prod x_j^{alpha_j-1}, (x_j)>=0, sum x_j=1

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
    Ghosal App G
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Finite-dimensional Dirichlet distribution: Dir(alpha_1..alpha_k) on simplex S_k"})


def cheatsheet():
    return "gh_ap_g1: Finite-dimensional Dirichlet distribution: Dir(alpha_1..alpha_k) on simplex S_k"
