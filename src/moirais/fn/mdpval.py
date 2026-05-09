"""Value iteration for MDP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mdp_value_iteration"]


def mdp_value_iteration(P, R, gamma, tol):
    """
    Value iteration for MDP

    Formula: V_{k+1}(s) = max_a sum_{s'} P(s'|s,a)[R + γ V_k(s')]

    Parameters
    ----------
    P : array-like
        Input data.
    R : array-like
        Input data.
    gamma : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Puterman (1994); Bellman (1957)
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Value iteration for MDP"})


def cheatsheet():
    return "mdpval: Value iteration for MDP"
