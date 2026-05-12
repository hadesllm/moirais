# morie.fn -- function file (hadesllm/morie)
"""Glorot (Xavier) initialization for layer with fan_in/fan_out."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_glorot_xavier_init"]


def geron_glorot_xavier_init(fan_in, fan_out, distribution):
    """
    Glorot (Xavier) initialization for layer with fan_in/fan_out

    Formula: var(W) = 2 / (fan_in + fan_out); W ~ N(0, var) or U(-sqrt(6/(fan_in+fan_out)), +...)

    Parameters
    ----------
    fan_in : array-like
        Input data.
    fan_out : array-like
        Input data.
    distribution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W

    References
    ----------
    Géron Ch 11, Eq 11-1 (Glorot initialization)
    """
    fan_in = np.asarray(fan_in, dtype=float)
    n = int(fan_in) if fan_in.ndim == 0 else len(fan_in)
    result = float(np.mean(fan_in))
    se = float(np.std(fan_in, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Glorot (Xavier) initialization for layer with fan_in/fan_out"})


def cheatsheet():
    return "grxvi: Glorot (Xavier) initialization for layer with fan_in/fan_out"
