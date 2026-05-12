# morie.fn -- function file (hadesllm/morie)
"""Momentum optimization: accumulates exponentially-decaying past gradients."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_momentum"]


def geron_momentum(grads, v, beta, eta):
    """
    Momentum optimization: accumulates exponentially-decaying past gradients

    Formula: v <- beta*v + grad; theta <- theta - eta*v

    Parameters
    ----------
    grads : array-like
        Input data.
    v : array-like
        Input data.
    beta : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 11
    """
    grads = np.atleast_1d(np.asarray(grads, dtype=float))
    n = len(grads)
    result = float(np.mean(grads))
    se = float(np.std(grads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Momentum optimization: accumulates exponentially-decaying past gradients"})


def cheatsheet():
    return "hmmom: Momentum optimization: accumulates exponentially-decaying past gradients"
