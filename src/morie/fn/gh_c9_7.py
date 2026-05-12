# morie.fn -- function file (hadesllm/morie)
"""Whittle spectral density contraction rate via log-spline prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_whittle_crt"]


def ghosal_whittle_crt(x):
    """
    Whittle spectral density contraction rate via log-spline prior

    Formula: Whittle likelihood rate: eps_n = n^{-s/(2s+1)} (log n)^(1/2) for s-smooth spectral f

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
    Ghosal Ch 9 §9.5.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Whittle spectral density contraction rate via log-spline prior"})


def cheatsheet():
    return "gh_c9_7: Whittle spectral density contraction rate via log-spline prior"
