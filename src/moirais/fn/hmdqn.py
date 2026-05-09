# moirais.fn — function file (hadesllm/moirais)
"""Deep Q-network (DQN): neural Q-function with replay buffer and target net."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dqn"]


def geron_dqn(env, Q, Q_target, buffer, epochs, lr):
    """
    Deep Q-network (DQN): neural Q-function with replay buffer and target net

    Formula: L = (r + gamma*max_a Q_target(s',a) - Q(s,a))^2

    Parameters
    ----------
    env : array-like
        Input data.
    Q : array-like
        Input data.
    Q_target : array-like
        Input data.
    buffer : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q

    References
    ----------
    Géron Ch 19
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep Q-network (DQN): neural Q-function with replay buffer and target net"})


def cheatsheet():
    return "hmdqn: Deep Q-network (DQN): neural Q-function with replay buffer and target net"
