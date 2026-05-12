# morie.fn -- function file (hadesllm/morie)
"""Reward function R(s, a, s')."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_reward_function"]


def geron_reward_function(s, a, s_next):
    """
    Reward function R(s, a, s')

    Formula: r_t = R(s_t, a_t, s_{t+1})

    Parameters
    ----------
    s : array-like
        Input data.
    a : array-like
        Input data.
    s_next : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: reward

    References
    ----------
    Géron Ch 19
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reward function R(s, a, s')"})


def cheatsheet():
    return "hmrwd: Reward function R(s, a, s')"
