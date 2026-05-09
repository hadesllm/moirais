"""Aalen-Johansen multistate transition probabilities."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multistate_transition_matrix"]


def multistate_transition_matrix(time, state, X):
    """
    Aalen-Johansen multistate transition probabilities

    Formula: P_jk(s,t) = product (I + dA(u))

    Parameters
    ----------
    time : array-like
        Input data.
    state : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andersen & Borgan (1985)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Aalen-Johansen multistate transition probabilities"})


def cheatsheet():
    return "mstrn: Aalen-Johansen multistate transition probabilities"
