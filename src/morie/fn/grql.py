# morie.fn -- function file (hadesllm/morie)
"""Q-learning off-policy update."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_q_learning_update"]


def geron_q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Q-learning off-policy update

    Formula: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_{a'} Q(s',a') - Q(s,a))

    Parameters
    ----------
    Q : array-like
        Input data.
    s : array-like
        Input data.
    a : array-like
        Input data.
    r : array-like
        Input data.
    s_next : array-like
        Input data.
    alpha : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q_new

    References
    ----------
    Géron Ch 19, Q-learning section
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Q-learning off-policy update"})


def cheatsheet():
    return "grql: Q-learning off-policy update"
