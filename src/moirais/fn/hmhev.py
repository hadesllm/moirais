# moirais.fn — function file (hadesllm/moirais)
"""Heaviside step activation function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_heaviside"]


def geron_heaviside(z):
    """
    Heaviside step activation function

    Formula: step(z) = 1 if z>=0 else 0

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 9
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heaviside step activation function"})


def cheatsheet():
    return "hmhev: Heaviside step activation function"
