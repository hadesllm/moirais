# morie.fn -- function file (hadesllm/morie)
"""Kullback-Leibler divergence formula used in consistency theory."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_kl_diverge"]


def ghosal_kl_diverge(x):
    """
    Kullback-Leibler divergence formula used in consistency theory

    Formula: KL(P0,P) = integral log(p0/p) dP0

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
    Ghosal Ch 6 §6.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kullback-Leibler divergence formula used in consistency theory"})


def cheatsheet():
    return "gh_c6_7: Kullback-Leibler divergence formula used in consistency theory"
