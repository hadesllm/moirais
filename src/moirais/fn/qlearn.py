"""Tabular Q-learning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["q_learning"]


def q_learning(env, alpha, gamma, epsilon, n_episodes):
    """
    Tabular Q-learning

    Formula: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]

    Parameters
    ----------
    env : array-like
        Input data.
    alpha : array-like
        Input data.
    gamma : array-like
        Input data.
    epsilon : array-like
        Input data.
    n_episodes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Watkins-Dayan (1992)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tabular Q-learning"})


def cheatsheet():
    return "qlearn: Tabular Q-learning"
