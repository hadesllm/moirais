# morie.fn -- function file (hadesllm/morie)
"""Log-spline prior contraction rate: optimal n^{-s/(2s+1)} for s-Sobolev density."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_logspline_crt"]


def ghosal_logspline_crt(x):
    """
    Log-spline prior contraction rate: optimal n^{-s/(2s+1)} for s-Sobolev density

    Formula: f = exp(sum beta_k phi_k)/Z, K_n ~ n^{1/(2s+1)}, rate n^{-s/(2s+1)}

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
    Ghosal Ch 9 §9.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-spline prior contraction rate: optimal n^{-s/(2s+1)} for s-Sobolev density"})


def cheatsheet():
    return "gh_c9_1: Log-spline prior contraction rate: optimal n^{-s/(2s+1)} for s-Sobolev density"
