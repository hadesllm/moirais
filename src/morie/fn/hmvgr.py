# morie.fn — function file (hadesllm/morie)
"""Vanishing gradients: small gradients shrink through many layers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_vanishing_gradients"]


def geron_vanishing_gradients(grads):
    """
    Vanishing gradients: small gradients shrink through many layers

    Formula: ||grad L^(l)|| -> 0 as l decreases when |phi'| < 1 repeatedly

    Parameters
    ----------
    grads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_vanishing

    References
    ----------
    Géron Ch 11
    """
    grads = np.atleast_1d(np.asarray(grads, dtype=float))
    n = len(grads)
    result = float(np.mean(grads))
    se = float(np.std(grads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vanishing gradients: small gradients shrink through many layers"})


def cheatsheet():
    return "hmvgr: Vanishing gradients: small gradients shrink through many layers"
