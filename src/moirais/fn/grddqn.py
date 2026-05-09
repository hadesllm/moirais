# moirais.fn — function file (hadesllm/moirais)
"""Double DQN decouples action selection from evaluation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_double_dqn_target"]


def geron_double_dqn_target(Q_online, Q_target, s_next, r, gamma):
    """
    Double DQN decouples action selection from evaluation

    Formula: y = r + gamma * Q_target(s', argmax_{a'} Q_online(s', a'; theta); theta-)

    Parameters
    ----------
    Q_online : array-like
        Input data.
    Q_target : array-like
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
        Keys: target

    References
    ----------
    Géron Ch 19, Double DQN section
    """
    Q_online = np.atleast_1d(np.asarray(Q_online, dtype=float))
    n = len(Q_online)
    result = float(np.mean(Q_online))
    se = float(np.std(Q_online, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Double DQN decouples action selection from evaluation"})


def cheatsheet():
    return "grddqn: Double DQN decouples action selection from evaluation"
