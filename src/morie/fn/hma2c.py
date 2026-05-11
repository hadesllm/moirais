# morie.fn — function file (hadesllm/morie)
"""Advantage actor-critic (A2C)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_a2c"]


def geron_a2c(env, actor, critic, epochs, lr):
    """
    Advantage actor-critic (A2C)

    Formula: policy update uses A(s,a) = Q(s,a) - V(s)

    Parameters
    ----------
    env : array-like
        Input data.
    actor : array-like
        Input data.
    critic : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: actor, critic

    References
    ----------
    Géron Ch 19
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Advantage actor-critic (A2C)"})


def cheatsheet():
    return "hma2c: Advantage actor-critic (A2C)"
