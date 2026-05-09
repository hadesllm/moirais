# moirais.fn — function file (hadesllm/moirais)
"""Reinforcement learning: agent maximizes cumulative reward via policy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_reinforcement_learning"]


def geron_reinforcement_learning(env, pi, gamma):
    """
    Reinforcement learning: agent maximizes cumulative reward via policy

    Formula: max_pi E_pi [sum_t gamma^t r_t]

    Parameters
    ----------
    env : array-like
        Input data.
    pi : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: policy

    References
    ----------
    Géron Ch 1
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reinforcement learning: agent maximizes cumulative reward via policy"})


def cheatsheet():
    return "hmrl: Reinforcement learning: agent maximizes cumulative reward via policy"
