# morie.fn — function file (hadesllm/morie)
"""Constrained Dirichlet process: DP conditioned on linear constraint integral f dG = c."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_constr_dp"]


def ghosal_constr_dp(x):
    """
    Constrained Dirichlet process: DP conditioned on linear constraint integral f dG = c

    Formula: DP(alpha, G0) | integral f dG = c

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
    Ghosal Ch 4 §4.6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Constrained Dirichlet process: DP conditioned on linear constraint integral f dG = c"})


def cheatsheet():
    return "gh_c4_22: Constrained Dirichlet process: DP conditioned on linear constraint integral f dG = c"
