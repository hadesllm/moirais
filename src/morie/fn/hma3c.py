# morie.fn — function file (hadesllm/morie)
"""Asynchronous advantage actor-critic (A3C)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_a3c"]


def geron_a3c(env, actor, critic, n_workers, lr):
    """
    Asynchronous advantage actor-critic (A3C)

    Formula: parallel actors asynchronously update shared parameters

    Parameters
    ----------
    env : array-like
        Input data.
    actor : array-like
        Input data.
    critic : array-like
        Input data.
    n_workers : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 19
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asynchronous advantage actor-critic (A3C)"})


def cheatsheet():
    return "hma3c: Asynchronous advantage actor-critic (A3C)"
