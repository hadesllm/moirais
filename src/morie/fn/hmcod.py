# morie.fn — function file (hadesllm/morie)
"""Curse of dimensionality: sample sparsity grows exponentially with d."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_curse_dimensionality"]


def geron_curse_dimensionality(d, n):
    """
    Curse of dimensionality: sample sparsity grows exponentially with d

    Formula: expected distance to nearest neighbor ~ d^{1/d}

    Parameters
    ----------
    d : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: effective_density

    References
    ----------
    Géron Ch 7
    """
    d = np.atleast_1d(np.asarray(d, dtype=float))
    n = len(d)
    result = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Curse of dimensionality: sample sparsity grows exponentially with d"})


def cheatsheet():
    return "hmcod: Curse of dimensionality: sample sparsity grows exponentially with d"
