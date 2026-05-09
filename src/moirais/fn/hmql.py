# moirais.fn — function file (hadesllm/moirais)
"""Q-learning: off-policy TD control."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_q_learning"]


def geron_q_learning(Q, s, a, r, s_next, alpha, gamma):
    """
    Q-learning: off-policy TD control

    Formula: Q(s,a) <- Q(s,a) + alpha*(r + gamma*max_{a'} Q(s',a') - Q(s,a))

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
        Keys: Q

    References
    ----------
    Géron Ch 19
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Q-learning: off-policy TD control"})


def cheatsheet():
    return "hmql: Q-learning: off-policy TD control"
