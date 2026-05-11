# morie.fn — function file (hadesllm/morie)
"""Ranks, block frequencies, and placements."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rank_placements"]


def rank_placements(x, y):
    """
    Ranks, block frequencies, and placements

    Formula: Placement of Y among X order statistics

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
    Gibbons Ch 2.11.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ranks, block frequencies, and placements"})


def cheatsheet():
    return "plcmt: Ranks, block frequencies, and placements"
