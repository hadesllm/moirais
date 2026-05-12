# morie.fn -- function file (hadesllm/morie)
"""Exploding gradients: gradients grow through layers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_exploding_gradients"]


def geron_exploding_gradients(grads):
    """
    Exploding gradients: gradients grow through layers

    Formula: ||grad L^(l)|| -> inf as l decreases when |phi'| > 1 repeatedly

    Parameters
    ----------
    grads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_exploding

    References
    ----------
    Géron Ch 11
    """
    grads = np.atleast_1d(np.asarray(grads, dtype=float))
    n = len(grads)
    result = float(np.mean(grads))
    se = float(np.std(grads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exploding gradients: gradients grow through layers"})


def cheatsheet():
    return "hmxgr: Exploding gradients: gradients grow through layers"
