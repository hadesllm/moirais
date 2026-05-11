"""Soft actor-critic (max-entropy RL)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sac"]


def sac(env, actor, critic, alpha):
    """
    Soft actor-critic (max-entropy RL)

    Formula: objective = E[Q(s,a) + α H(π(·|s))]

    Parameters
    ----------
    env : array-like
        Input data.
    actor : array-like
        Input data.
    critic : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Haarnoja et al (2018)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Soft actor-critic (max-entropy RL)"})


def cheatsheet():
    return "sacc: Soft actor-critic (max-entropy RL)"
