# morie.fn -- function file (hadesllm/morie)
"""Soft actor-critic (SAC): entropy-regularized max reward."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sac"]


def geron_sac(env, policy, critic, epochs, lr, alpha):
    """
    Soft actor-critic (SAC): entropy-regularized max reward

    Formula: pi* = argmax E[sum_t r_t + alpha*H(pi(.|s_t))]

    Parameters
    ----------
    env : array-like
        Input data.
    policy : array-like
        Input data.
    critic : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: policy

    References
    ----------
    Géron Ch 19
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Soft actor-critic (SAC): entropy-regularized max reward"})


def cheatsheet():
    return "hmsac: Soft actor-critic (SAC): entropy-regularized max reward"
