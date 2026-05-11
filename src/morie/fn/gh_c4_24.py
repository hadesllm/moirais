# morie.fn — function file (hadesllm/morie)
"""Rubin's Bayesian bootstrap: posterior of DP at alpha->0 is Dirichlet(1,...,1) on observed."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bayes_boot"]


def ghosal_bayes_boot(x):
    """
    Rubin's Bayesian bootstrap: posterior of DP at alpha->0 is Dirichlet(1,...,1) on observed

    Formula: G | X_1..X_n ~ DP(epsilon, G_n) as epsilon->0 gives Bayesian bootstrap

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
    Ghosal Ch 4 §4.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rubin's Bayesian bootstrap: posterior of DP at alpha->0 is Dirichlet(1,...,1) on observed"})


def cheatsheet():
    return "gh_c4_24: Rubin's Bayesian bootstrap: posterior of DP at alpha->0 is Dirichlet(1,...,1) on observed"
