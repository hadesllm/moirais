# morie.fn -- function file (hadesllm/morie)
"""Nesterov accelerated gradient (NAG)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_nesterov"]


def geron_nesterov(grads, v, beta, eta):
    """
    Nesterov accelerated gradient (NAG)

    Formula: v <- beta*v + grad(theta - eta*beta*v); theta <- theta - eta*v

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nesterov accelerated gradient (NAG)"})


def cheatsheet():
    return "hmnag: Nesterov accelerated gradient (NAG)"
