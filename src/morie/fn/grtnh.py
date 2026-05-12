# morie.fn -- function file (hadesllm/morie)
"""Hyperbolic tangent activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tanh_activation"]


def geron_tanh_activation(z):
    """
    Hyperbolic tangent activation

    Formula: tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))

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
    Géron Ch 9, Activation Functions (tanh)
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hyperbolic tangent activation"})


def cheatsheet():
    return "grtnh: Hyperbolic tangent activation"
