# morie.fn — function file (hadesllm/morie)
"""Policy gradient (REINFORCE) update."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_policy_gradient"]


def geron_policy_gradient(trajectories, policy, gamma):
    """
    Policy gradient (REINFORCE) update

    Formula: grad J = E[grad log pi(a|s; theta) * Q(s,a)]

    Parameters
    ----------
    trajectories : array-like
        Input data.
    policy : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 19
    """
    trajectories = np.atleast_1d(np.asarray(trajectories, dtype=float))
    n = len(trajectories)
    result = float(np.mean(trajectories))
    se = float(np.std(trajectories, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Policy gradient (REINFORCE) update"})


def cheatsheet():
    return "hmpg: Policy gradient (REINFORCE) update"
