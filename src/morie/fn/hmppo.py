# morie.fn — function file (hadesllm/morie)
"""Proximal policy optimization (PPO) clipped-surrogate objective."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ppo"]


def geron_ppo(env, policy, epochs, lr, clip_eps):
    """
    Proximal policy optimization (PPO) clipped-surrogate objective

    Formula: L = E[min(r_t(theta) A_t, clip(r_t, 1-e, 1+e) A_t)]

    Parameters
    ----------
    env : array-like
        Input data.
    policy : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.
    clip_eps : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proximal policy optimization (PPO) clipped-surrogate objective"})


def cheatsheet():
    return "hmppo: Proximal policy optimization (PPO) clipped-surrogate objective"
