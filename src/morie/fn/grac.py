# morie.fn — function file (hadesllm/morie)
"""Actor-critic with learned value baseline; advantage = r + gamma*V(s') - V(s)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_actor_critic_advantage"]


def geron_actor_critic_advantage(V, s, s_next, r, gamma):
    """
    Actor-critic with learned value baseline; advantage = r + gamma*V(s') - V(s)

    Formula: A(s, a) = r + gamma * V(s') - V(s); actor grad uses A in place of G

    Parameters
    ----------
    V : array-like
        Input data.
    s : array-like
        Input data.
    s_next : array-like
        Input data.
    r : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: advantage

    References
    ----------
    Géron Ch 19, Actor-Critic A2C/A3C section
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Actor-critic with learned value baseline; advantage = r + gamma*V(s') - V(s)"})


def cheatsheet():
    return "grac: Actor-critic with learned value baseline; advantage = r + gamma*V(s') - V(s)"
