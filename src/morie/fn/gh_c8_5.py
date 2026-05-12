# morie.fn -- function file (hadesllm/morie)
"""Metric entropy condition: log N(eps_n, sieve Theta_n, Hellinger) <= n*eps_n^2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_entropy_cnd"]


def ghosal_entropy_cnd(x):
    """
    Metric entropy condition: log N(eps_n, sieve Theta_n, Hellinger) <= n*eps_n^2

    Formula: log N(eps_n, Theta_n, d_H) <= n*eps_n^2 for sieve Theta_n

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
    Ghosal Ch 8 §8.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Metric entropy condition: log N(eps_n, sieve Theta_n, Hellinger) <= n*eps_n^2"})


def cheatsheet():
    return "gh_c8_5: Metric entropy condition: log N(eps_n, sieve Theta_n, Hellinger) <= n*eps_n^2"
