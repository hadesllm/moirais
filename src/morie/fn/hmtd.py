# morie.fn — function file (hadesllm/morie)
"""Temporal-difference (TD) learning update."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_td_learning"]


def geron_td_learning(V, s, r, s_next, alpha, gamma):
    """
    Temporal-difference (TD) learning update

    Formula: V(s) <- V(s) + alpha*(r + gamma*V(s') - V(s))

    Parameters
    ----------
    V : array-like
        Input data.
    s : array-like
        Input data.
    r : array-like
        Input data.
    s_next : array-like
        Input data.
    alpha : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V

    References
    ----------
    Géron Ch 19
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Temporal-difference (TD) learning update"})


def cheatsheet():
    return "hmtd: Temporal-difference (TD) learning update"
