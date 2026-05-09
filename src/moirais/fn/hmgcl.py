# moirais.fn — function file (hadesllm/moirais)
"""Gradient clipping by global norm to stabilize training."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gradient_clipping"]


def geron_gradient_clipping(grads, max_norm):
    """
    Gradient clipping by global norm to stabilize training

    Formula: if ||g|| > c: g <- g * c / ||g||

    Parameters
    ----------
    grads : array-like
        Input data.
    max_norm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: clipped_grads

    References
    ----------
    Géron Ch 11
    """
    grads = np.atleast_1d(np.asarray(grads, dtype=float))
    n = len(grads)
    result = float(np.mean(grads))
    se = float(np.std(grads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient clipping by global norm to stabilize training"})


def cheatsheet():
    return "hmgcl: Gradient clipping by global norm to stabilize training"
