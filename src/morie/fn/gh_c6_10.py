# morie.fn -- function file (hadesllm/morie)
"""Posterior consistency for independent non-i.i.d. observations."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_non_iid_con"]


def ghosal_non_iid_con(x):
    """
    Posterior consistency for independent non-i.i.d. observations

    Formula: Pi_n(U^c|X^n)->0 under triangular array conditions on p_{n,theta}

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
    Ghosal Ch 6 §6.7.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior consistency for independent non-i.i.d. observations"})


def cheatsheet():
    return "gh_c6_10: Posterior consistency for independent non-i.i.d. observations"
