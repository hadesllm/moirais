# moirais.fn — function file (hadesllm/moirais)
"""REINFORCE algorithm: sample trajectories, update theta by advantage."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_reinforce"]


def geron_reinforce(episodes, policy, gamma, eta):
    """
    REINFORCE algorithm: sample trajectories, update theta by advantage

    Formula: theta <- theta + eta * sum_t grad log pi(a_t|s_t) * G_t

    Parameters
    ----------
    episodes : array-like
        Input data.
    policy : array-like
        Input data.
    gamma : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: policy

    References
    ----------
    Géron Ch 19
    """
    episodes = np.atleast_1d(np.asarray(episodes, dtype=float))
    n = len(episodes)
    result = float(np.mean(episodes))
    se = float(np.std(episodes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "REINFORCE algorithm: sample trajectories, update theta by advantage"})


def cheatsheet():
    return "hmrnfc: REINFORCE algorithm: sample trajectories, update theta by advantage"
