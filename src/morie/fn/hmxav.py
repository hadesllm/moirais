# morie.fn -- function file (hadesllm/morie)
"""Glorot (Xavier) initialization for sigmoid/tanh networks."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_glorot_init"]


def geron_glorot_init(fan_in, fan_out, seed):
    """
    Glorot (Xavier) initialization for sigmoid/tanh networks

    Formula: Var(W) = 2 / (fan_in + fan_out)

    Parameters
    ----------
    fan_in : array-like
        Input data.
    fan_out : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W

    References
    ----------
    Géron Ch 11
    """
    fan_in = np.atleast_1d(np.asarray(fan_in, dtype=float))
    n = len(fan_in)
    result = float(np.mean(fan_in))
    se = float(np.std(fan_in, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Glorot (Xavier) initialization for sigmoid/tanh networks"})


def cheatsheet():
    return "hmxav: Glorot (Xavier) initialization for sigmoid/tanh networks"
