# moirais.fn — function file (hadesllm/moirais)
"""Epsilon-greedy action selection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_epsilon_greedy"]


def geron_epsilon_greedy(Q_s, eps, seed):
    """
    Epsilon-greedy action selection

    Formula: a = argmax_a Q(s,a) with prob 1-eps; else uniform random a

    Parameters
    ----------
    Q_s : array-like
        Input data.
    eps : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: action

    References
    ----------
    Géron Ch 19, Epsilon-Greedy Exploration section
    """
    Q_s = np.atleast_1d(np.asarray(Q_s, dtype=float))
    n = len(Q_s)
    result = float(np.mean(Q_s))
    se = float(np.std(Q_s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Epsilon-greedy action selection"})


def cheatsheet():
    return "grepl: Epsilon-greedy action selection"
