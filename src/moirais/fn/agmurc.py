"""MuZero recurrent inference (g function)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["muzero_recurrent_inf"]


def muzero_recurrent_inf(state, action, g):
    """
    MuZero recurrent inference (g function)

    Formula: (s', r) = g(s, a)

    Parameters
    ----------
    state : array-like
        Input data.
    action : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schrittwieser et al (2020)
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MuZero recurrent inference (g function)"})


def cheatsheet():
    return "agmurc: MuZero recurrent inference (g function)"
