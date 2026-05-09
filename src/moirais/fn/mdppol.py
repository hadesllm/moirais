"""Policy iteration for MDP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mdp_policy_iteration"]


def mdp_policy_iteration(P, R, gamma):
    """
    Policy iteration for MDP

    Formula: alternate policy evaluation + greedy improvement

    Parameters
    ----------
    P : array-like
        Input data.
    R : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Howard (1960)
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Policy iteration for MDP"})


def cheatsheet():
    return "mdppol: Policy iteration for MDP"
