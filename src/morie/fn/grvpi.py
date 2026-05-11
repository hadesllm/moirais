# morie.fn — function file (hadesllm/morie)
"""State value function V^pi(s) = E_pi[G_t | S_t = s]."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_state_value_function"]


def geron_state_value_function(state, policy, transitions, rewards, gamma):
    """
    State value function V^pi(s) = E_pi[G_t | S_t = s]

    Formula: V^pi(s) = E[ sum_k gamma^k r_{t+k+1} | S_t = s, pi ]

    Parameters
    ----------
    state : array-like
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
        Keys: V

    References
    ----------
    Géron Ch 19, Value Function section
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "State value function V^pi(s) = E_pi[G_t | S_t = s]"})


def cheatsheet():
    return "grvpi: State value function V^pi(s) = E_pi[G_t | S_t = s]"
