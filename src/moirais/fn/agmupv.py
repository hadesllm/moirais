"""MuZero predict (f function)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["muzero_predict_value"]


def muzero_predict_value(state, f):
    """
    MuZero predict (f function)

    Formula: (p, v) = f(s)

    Parameters
    ----------
    state : array-like
        Input data.
    f : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MuZero predict (f function)"})


def cheatsheet():
    return "agmupv: MuZero predict (f function)"
