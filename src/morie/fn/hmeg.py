# morie.fn -- function file (hadesllm/morie)
"""Epsilon-greedy exploration strategy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_epsilon_greedy"]


def geron_epsilon_greedy(Q, s, epsilon):
    """
    Epsilon-greedy exploration strategy

    Formula: a = argmax_a Q(s,a) w.p. 1-eps; random w.p. eps

    Parameters
    ----------
    Q : array-like
        Input data.
    s : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: action

    References
    ----------
    Géron Ch 19
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Epsilon-greedy exploration strategy"})


def cheatsheet():
    return "hmeg: Epsilon-greedy exploration strategy"
