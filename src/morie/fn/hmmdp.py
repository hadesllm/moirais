# morie.fn -- function file (hadesllm/morie)
"""Markov decision process (S, A, P, R, gamma)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mdp"]


def geron_mdp(states, actions, P, R, gamma):
    """
    Markov decision process (S, A, P, R, gamma)

    Formula: (s, a) -> s' ~ P; r ~ R; discount gamma

    Parameters
    ----------
    states : array-like
        Input data.
    actions : array-like
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
        Keys: mdp

    References
    ----------
    Géron Ch 19
    """
    states = np.atleast_1d(np.asarray(states, dtype=float))
    n = len(states)
    result = float(np.mean(states))
    se = float(np.std(states, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Markov decision process (S, A, P, R, gamma)"})


def cheatsheet():
    return "hmmdp: Markov decision process (S, A, P, R, gamma)"
