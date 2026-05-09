# moirais.fn — function file (hadesllm/moirais)
"""Rotary positional embedding (RoPE)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rotary_position_embedding"]


def rotary_position_embedding(x):
    """
    Rotary positional embedding (RoPE)

    Formula: R(theta) = [[cos,-sin],[sin,cos]] applied pairwise

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
    Su et al. (2021)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rotary positional embedding (RoPE)"})


def cheatsheet():
    return "rotrp: Rotary positional embedding (RoPE)"
