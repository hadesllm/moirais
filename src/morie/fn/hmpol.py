# morie.fn — function file (hadesllm/morie)
"""Policy pi(a|s) or deterministic pi(s)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_policy"]


def geron_policy(state, pi):
    """
    Policy pi(a|s) or deterministic pi(s)

    Formula: a_t ~ pi(. | s_t)

    Parameters
    ----------
    state : array-like
        Input data.
    pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: action

    References
    ----------
    Géron Ch 19
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Policy pi(a|s) or deterministic pi(s)"})


def cheatsheet():
    return "hmpol: Policy pi(a|s) or deterministic pi(s)"
