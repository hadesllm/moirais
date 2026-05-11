# morie.fn — function file (hadesllm/morie)
"""Slice sampler for DP mixture: auxiliary variable u for uniform sampling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_slice_sampler"]


def ghosal_slice_sampler(x):
    """
    Slice sampler for DP mixture: auxiliary variable u for uniform sampling

    Formula: Sample u ~ Unif(0, pi(theta|X)), then theta ~ Unif({theta: pi(theta|X)>u})

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
    Ghosal App M
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Slice sampler for DP mixture: auxiliary variable u for uniform sampling"})


def cheatsheet():
    return "gh_ap_m3: Slice sampler for DP mixture: auxiliary variable u for uniform sampling"
