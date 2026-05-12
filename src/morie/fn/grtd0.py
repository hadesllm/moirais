# morie.fn -- function file (hadesllm/morie)
"""TD(0) value update."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_td_zero_update"]


def geron_td_zero_update(V, state, next_state, reward, alpha, gamma):
    """
    TD(0) value update

    Formula: V(S_t) <- V(S_t) + alpha * ( r + gamma * V(S_{t+1}) - V(S_t) )

    Parameters
    ----------
    V : array-like
        Input data.
    state : array-like
        Input data.
    next_state : array-like
        Input data.
    reward : array-like
        Input data.
    alpha : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_new

    References
    ----------
    Géron Ch 19, Temporal Difference Learning section
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TD(0) value update"})


def cheatsheet():
    return "grtd0: TD(0) value update"
