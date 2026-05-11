# morie.fn — function file (hadesllm/morie)
"""Heaviside step activation function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_heaviside_step"]


def geron_heaviside_step(z):
    """
    Heaviside step activation function

    Formula: heaviside(z) = 1 if z >= 0 else 0

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: step

    References
    ----------
    Géron Ch 9, Heaviside step activation
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heaviside step activation function"})


def cheatsheet():
    return "grhev: Heaviside step activation function"
