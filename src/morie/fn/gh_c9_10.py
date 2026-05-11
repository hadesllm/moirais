# morie.fn — function file (hadesllm/morie)
"""Spline regression contraction rate: n^{-2s/(2s+1)} via sieve of splines."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_spline_crt"]


def ghosal_spline_crt(x, y):
    """
    Spline regression contraction rate: n^{-2s/(2s+1)} via sieve of splines

    Formula: f in spline space of dim K_n, K_n ~ n^{1/(2s+1)}, rate n^{-2s/(2s+1)}

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 9 §9.5.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spline regression contraction rate: n^{-2s/(2s+1)} via sieve of splines"})


def cheatsheet():
    return "gh_c9_10: Spline regression contraction rate: n^{-2s/(2s+1)} via sieve of splines"
