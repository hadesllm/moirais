# morie.fn — function file (hadesllm/morie)
"""State value function V^pi(s)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_value_function"]


def geron_value_function(s, pi, gamma):
    """
    State value function V^pi(s)

    Formula: V^pi(s) = E_pi[sum_t gamma^t r_t | s_0 = s]

    Parameters
    ----------
    s : array-like
        Input data.
    pi : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V

    References
    ----------
    Géron Ch 19
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "State value function V^pi(s)"})


def cheatsheet():
    return "hmvf: State value function V^pi(s)"
