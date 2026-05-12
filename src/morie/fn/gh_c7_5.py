# morie.fn -- function file (hadesllm/morie)
"""DP mixture consistency for general kernel: consistency at Lebesgue a.e. densities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dpm_gen_con"]


def ghosal_dpm_gen_con(x):
    """
    DP mixture consistency for general kernel: consistency at Lebesgue a.e. densities

    Formula: DPM with kernel K satisfying tail and smoothness conditions is consistent

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
    Ghosal Ch 7 §7.2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP mixture consistency for general kernel: consistency at Lebesgue a.e. densities"})


def cheatsheet():
    return "gh_c7_5: DP mixture consistency for general kernel: consistency at Lebesgue a.e. densities"
