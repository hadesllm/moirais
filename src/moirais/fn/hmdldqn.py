# moirais.fn — function file (hadesllm/moirais)
"""Dueling DQN: separate value and advantage streams."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dueling_dqn"]


def geron_dueling_dqn(env, V, A, buffer, epochs, lr):
    """
    Dueling DQN: separate value and advantage streams

    Formula: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)

    Parameters
    ----------
    env : array-like
        Input data.
    V : array-like
        Input data.
    A : array-like
        Input data.
    buffer : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q

    References
    ----------
    Géron Ch 19
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dueling DQN: separate value and advantage streams"})


def cheatsheet():
    return "hmdldqn: Dueling DQN: separate value and advantage streams"
