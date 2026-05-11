# morie.fn — function file (hadesllm/morie)
"""Swish / SiLU activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_swish"]


def geron_swish(z):
    """
    Swish / SiLU activation

    Formula: swish(z) = z * sigmoid(z)

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a

    References
    ----------
    Géron Ch 11
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Swish / SiLU activation"})


def cheatsheet():
    return "hmswi: Swish / SiLU activation"
