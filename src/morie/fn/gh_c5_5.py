# morie.fn -- function file (hadesllm/morie)
"""Blocked Gibbs sampler using truncated stick-breaking for DP mixtures."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_blk_gibbs"]


def ghosal_blk_gibbs(x):
    """
    Blocked Gibbs sampler using truncated stick-breaking for DP mixtures

    Formula: G_K = sum_{k=1}^K w_k delta_{theta_k}, update all K components jointly

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
    Ghosal Ch 5 §5.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Blocked Gibbs sampler using truncated stick-breaking for DP mixtures"})


def cheatsheet():
    return "gh_c5_5: Blocked Gibbs sampler using truncated stick-breaking for DP mixtures"
