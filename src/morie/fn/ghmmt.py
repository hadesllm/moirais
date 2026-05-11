# morie.fn — function file (hadesllm/morie)
"""Moment matching for DP mixtures."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_moment_matching"]


def ghosal_moment_matching(x):
    """
    Moment matching for DP mixtures

    Formula: E[G(A)] = G0(A), Var[G(A)] = G0(A)(1-G0(A))/(alpha+1)

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
    Ghosal Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moment matching for DP mixtures"})


def cheatsheet():
    return "ghmmt: Moment matching for DP mixtures"
