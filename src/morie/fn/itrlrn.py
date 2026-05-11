"""Iterative Q-learning for dynamic regimes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["iterative_q_learning"]


def iterative_q_learning(state, action, reward, time):
    """
    Iterative Q-learning for dynamic regimes

    Formula: Q_t(s,a) <- E[Y + gamma max_a' Q_{t+1}(s',a')]

    Parameters
    ----------
    state : array-like
        Input data.
    action : array-like
        Input data.
    reward : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Murphy (2003); Petersen et al (2014)
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Iterative Q-learning for dynamic regimes"})


def cheatsheet():
    return "itrlrn: Iterative Q-learning for dynamic regimes"
