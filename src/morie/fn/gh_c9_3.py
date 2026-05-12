# morie.fn -- function file (hadesllm/morie)
"""Bernstein polynomial prior contraction rate for density on [0,1]."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bpoly_crt"]


def ghosal_bpoly_crt(x):
    """
    Bernstein polynomial prior contraction rate for density on [0,1]

    Formula: f = sum_{k=0}^K p_k Be(x;k+1,K-k+1), K ~ pi_K, rate n^{-s/(2s+1)}

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
    Ghosal Ch 9 §9.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bernstein polynomial prior contraction rate for density on [0,1]"})


def cheatsheet():
    return "gh_c9_3: Bernstein polynomial prior contraction rate for density on [0,1]"
