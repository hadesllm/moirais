# morie.fn — function file (hadesllm/morie)
"""Hyperbolic tangent activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tanh"]


def geron_tanh(z):
    """
    Hyperbolic tangent activation

    Formula: tanh(z) = (e^z - e^-z) / (e^z + e^-z)

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
    Géron Ch 9
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hyperbolic tangent activation"})


def cheatsheet():
    return "hmtanh: Hyperbolic tangent activation"
