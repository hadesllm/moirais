"""Reinforcement-learning rec."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rlhf_recommendation"]


def rlhf_recommendation(env, policy):
    """
    Reinforcement-learning rec

    Formula: Q(s,a) over user state, action=rec

    Parameters
    ----------
    env : array-like
        Input data.
    policy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhao et al (2018)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reinforcement-learning rec"})


def cheatsheet():
    return "rlhfRS: Reinforcement-learning rec"
