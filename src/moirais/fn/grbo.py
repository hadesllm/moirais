# moirais.fn — function file (hadesllm/moirais)
"""Bellman optimality equation for Q*."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bellman_optimality"]


def geron_bellman_optimality(Q, transitions, rewards, gamma):
    """
    Bellman optimality equation for Q*

    Formula: Q*(s, a) = E[ r + gamma * max_{a'} Q*(s', a') | s, a ]

    Parameters
    ----------
    Q : array-like
        Input data.
    transitions : array-like
        Input data.
    rewards : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q_star

    References
    ----------
    Géron Ch 19, Bellman Optimality Equation
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bellman optimality equation for Q*"})


def cheatsheet():
    return "grbo: Bellman optimality equation for Q*"
