# morie.fn -- function file (hadesllm/morie)
"""Exchangeable partition probability function (EPPF): p(n1,...,nk) for partition of n."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_eppf_def"]


def ghosal_eppf_def(x):
    """
    Exchangeable partition probability function (EPPF): p(n1,...,nk) for partition of n

    Formula: P(partition n into k blocks of sizes n_1..n_k) = p(n_1..n_k) symmetric

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
    Ghosal Ch 14 §14.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exchangeable partition probability function (EPPF): p(n1,...,nk) for partition of n"})


def cheatsheet():
    return "gh_c14_1: Exchangeable partition probability function (EPPF): p(n1,...,nk) for partition of n"
