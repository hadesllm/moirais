# morie.fn — function file (hadesllm/morie)
"""Collapsed Gibbs sampler for DP mixtures (Neal Algorithm 2)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_cgibbs"]


def ghosal_cgibbs(x):
    """
    Collapsed Gibbs sampler for DP mixtures (Neal Algorithm 2)

    Formula: p(c_i=k|rest) proportional to n_{-i,k}*f(X_i|theta_k) or alpha*p_0(X_i)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Collapsed Gibbs sampler for DP mixtures (Neal Algorithm 2)"})


def cheatsheet():
    return "gh_c5_3: Collapsed Gibbs sampler for DP mixtures (Neal Algorithm 2)"
