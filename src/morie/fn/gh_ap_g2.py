# morie.fn — function file (hadesllm/morie)
"""Dirichlet moments: E[X_j]=alpha_j/alpha_0, Var[X_j]=alpha_j(alpha_0-alpha_j)/(alpha_0^2*(alpha_0+1))."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dir_moments"]


def ghosal_dir_moments(x):
    """
    Dirichlet moments: E[X_j]=alpha_j/alpha_0, Var[X_j]=alpha_j(alpha_0-alpha_j)/(alpha_0^2*(alpha_0+1))

    Formula: E[X_j] = alpha_j/alpha_0, Cov[X_i,X_j] = -alpha_i*alpha_j/(alpha_0^2*(alpha_0+1)) for i!=j

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet moments: E[X_j]=alpha_j/alpha_0, Var[X_j]=alpha_j(alpha_0-alpha_j)/(alpha_0^2*(alpha_0+1))"})


def cheatsheet():
    return "gh_ap_g2: Dirichlet moments: E[X_j]=alpha_j/alpha_0, Var[X_j]=alpha_j(alpha_0-alpha_j)/(alpha_0^2*(alpha_0+1))"
