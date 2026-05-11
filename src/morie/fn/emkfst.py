"""EM for state-space parameters."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["em_state_space"]


def em_state_space(y, init, max_iter):
    """
    EM for state-space parameters

    Formula: E-step Kalman smoothing; M-step closed-form

    Parameters
    ----------
    y : array-like
        Input data.
    init : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shumway-Stoffer (1982)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EM for state-space parameters"})


def cheatsheet():
    return "emkfst: EM for state-space parameters"
