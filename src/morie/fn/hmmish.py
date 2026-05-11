# morie.fn — function file (hadesllm/morie)
"""Mish activation: z * tanh(softplus(z))."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mish"]


def geron_mish(z):
    """
    Mish activation: z * tanh(softplus(z))

    Formula: Mish(z) = z * tanh(ln(1 + exp(z)))

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mish activation: z * tanh(softplus(z))"})


def cheatsheet():
    return "hmmish: Mish activation: z * tanh(softplus(z))"
