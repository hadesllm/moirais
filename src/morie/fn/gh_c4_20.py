# morie.fn -- function file (hadesllm/morie)
"""Mixtures of Dirichlet processes: prior is a mixture of DP(alpha, G0) over base measures."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_mix_dp"]


def ghosal_mix_dp(x):
    """
    Mixtures of Dirichlet processes: prior is a mixture of DP(alpha, G0) over base measures

    Formula: G | G0 ~ DP(alpha, G0), G0 ~ H => marginal is mixture of DP

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
    Ghosal Ch 4 §4.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixtures of Dirichlet processes: prior is a mixture of DP(alpha, G0) over base measures"})


def cheatsheet():
    return "gh_c4_20: Mixtures of Dirichlet processes: prior is a mixture of DP(alpha, G0) over base measures"
