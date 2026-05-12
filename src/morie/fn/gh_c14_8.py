# morie.fn -- function file (hadesllm/morie)
"""Gibbs-type partition process: EPPF of form prod V(n_j, k) / W(n, k)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gibbs_proc"]


def ghosal_gibbs_proc(x):
    """
    Gibbs-type partition process: EPPF of form prod V(n_j, k) / W(n, k)

    Formula: p(n_1..n_k) = C_n * prod_{j=1}^k V_{n_j} for Gibbs-type process

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
    Ghosal Ch 14 §14.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gibbs-type partition process: EPPF of form prod V(n_j, k) / W(n, k)"})


def cheatsheet():
    return "gh_c14_8: Gibbs-type partition process: EPPF of form prod V(n_j, k) / W(n, k)"
