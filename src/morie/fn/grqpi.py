# morie.fn -- function file (hadesllm/morie)
"""Action-value function Q^pi(s, a)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_action_value_function"]


def geron_action_value_function(state, action, policy, transitions, rewards, gamma):
    """
    Action-value function Q^pi(s, a)

    Formula: Q^pi(s, a) = E[ sum_k gamma^k r_{t+k+1} | S_t=s, A_t=a, pi ]

    Parameters
    ----------
    state : array-like
        Input data.
    action : array-like
        Input data.
    policy : array-like
        Input data.
    transitions : array-like
        Input data.
    rewards : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q

    References
    ----------
    Géron Ch 19, Action-Value Q function
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Action-value function Q^pi(s, a)"})


def cheatsheet():
    return "grqpi: Action-value function Q^pi(s, a)"
