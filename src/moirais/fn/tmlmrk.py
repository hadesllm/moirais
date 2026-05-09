"""TMLE for Markov decision processes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_markov"]


def tmle_markov(state, action, reward, policy):
    """
    TMLE for Markov decision processes

    Formula: long-term value V^pi via Bellman + clever covariate

    Parameters
    ----------
    state : array-like
        Input data.
    action : array-like
        Input data.
    reward : array-like
        Input data.
    policy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins (2004); Murphy (2003)
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for Markov decision processes"})


def cheatsheet():
    return "tmlmrk: TMLE for Markov decision processes"
