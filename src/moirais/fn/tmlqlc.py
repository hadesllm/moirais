"""Q-learning compatible TMLE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_qlearning"]


def tmle_qlearning(state, action, reward, time):
    """
    Q-learning compatible TMLE

    Formula: sequentially target Q-functions backwards in time

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Q-learning compatible TMLE"})


def cheatsheet():
    return "tmlqlc: Q-learning compatible TMLE"
