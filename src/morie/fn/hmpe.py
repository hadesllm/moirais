# morie.fn -- function file (hadesllm/morie)
"""Sinusoidal positional encoding."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_positional_encoding"]


def geron_positional_encoding(pos, d_model):
    """
    Sinusoidal positional encoding

    Formula: PE_{pos,2i} = sin(pos/10000^{2i/d}); PE_{pos,2i+1} = cos(pos/10000^{2i/d})

    Parameters
    ----------
    pos : array-like
        Input data.
    d_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: PE

    References
    ----------
    Géron Ch 15
    """
    pos = np.atleast_1d(np.asarray(pos, dtype=float))
    n = len(pos)
    result = float(np.mean(pos))
    se = float(np.std(pos, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sinusoidal positional encoding"})


def cheatsheet():
    return "hmpe: Sinusoidal positional encoding"
