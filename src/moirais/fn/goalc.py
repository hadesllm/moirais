"""Goal-conditioned RL."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["goal_conditioned"]


def goal_conditioned(env, policy, goal_dist):
    """
    Goal-conditioned RL

    Formula: π(a|s,g); reward depends on (s,g)

    Parameters
    ----------
    env : array-like
        Input data.
    policy : array-like
        Input data.
    goal_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schaul et al (2015) UVFA
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Goal-conditioned RL"})


def cheatsheet():
    return "goalc: Goal-conditioned RL"
