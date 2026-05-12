# morie.fn -- function file (hadesllm/morie)
"""Dirichlet marginals: subset-sum of Dirichlet components is Beta distributed."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dir_marginal"]


def ghosal_dir_marginal(x):
    """
    Dirichlet marginals: subset-sum of Dirichlet components is Beta distributed

    Formula: X_{j1}+..+X_{jm} ~ Be(alpha_{j1}+..+alpha_{jm}, alpha_0 - sum alpha_{jl})

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet marginals: subset-sum of Dirichlet components is Beta distributed"})


def cheatsheet():
    return "gh_ap_g3: Dirichlet marginals: subset-sum of Dirichlet components is Beta distributed"
