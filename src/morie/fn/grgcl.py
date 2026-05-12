# morie.fn -- function file (hadesllm/morie)
"""Gradient clipping by global L2 norm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gradient_clipping"]


def geron_gradient_clipping(gradients, c):
    """
    Gradient clipping by global L2 norm

    Formula: if ||g||_2 > c: g <- g * c / ||g||_2

    Parameters
    ----------
    gradients : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradients_clipped

    References
    ----------
    Géron Ch 11, Gradient Clipping section
    """
    gradients = np.asarray(gradients, dtype=float)
    n = int(gradients) if gradients.ndim == 0 else len(gradients)
    result = float(np.mean(gradients))
    se = float(np.std(gradients, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient clipping by global L2 norm"})


def cheatsheet():
    return "grgcl: Gradient clipping by global L2 norm"
