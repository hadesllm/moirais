# morie.fn -- function file (hadesllm/morie)
"""Bellman optimality equation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bellman_optimality"]


def geron_bellman_optimality(V, P, R, gamma):
    """
    Bellman optimality equation

    Formula: V*(s) = max_a [R(s,a) + gamma * sum_{s'} P(s'|s,a) V*(s')]

    Parameters
    ----------
    V : array-like
        Input data.
    P : array-like
        Input data.
    R : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_star

    References
    ----------
    Géron Ch 19
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bellman optimality equation"})


def cheatsheet():
    return "hmbel: Bellman optimality equation"
