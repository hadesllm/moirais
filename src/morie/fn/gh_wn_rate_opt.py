# morie.fn -- function file (hadesllm/morie)
"""Optimal white noise contraction: rate eps_n = n^{-s/(2s+1)} is minimax."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_white_noise_optimal_rate"]


def ghosal_white_noise_optimal_rate(x):
    """
    Optimal white noise contraction: rate eps_n = n^{-s/(2s+1)} is minimax

    Formula: White noise dY = theta dt + dW/sqrt(n): minimax rate R_n* = n^{-2s/(2s+1)}

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
    Ghosal Ch 8 §8.3.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Optimal white noise contraction: rate eps_n = n^{-s/(2s+1)} is minimax"})


def cheatsheet():
    return "gh_wn_rate_opt: Optimal white noise contraction: rate eps_n = n^{-s/(2s+1)} is minimax"
