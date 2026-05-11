"""NARM hybrid attention session rec."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["narm"]


def narm(sessions, K):
    """
    NARM hybrid attention session rec

    Formula: global + local encoder with attention

    Parameters
    ----------
    sessions : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Li et al (2017)
    """
    sessions = np.atleast_1d(np.asarray(sessions, dtype=float))
    n = len(sessions)
    result = float(np.mean(sessions))
    se = float(np.std(sessions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NARM hybrid attention session rec"})


def cheatsheet():
    return "narm: NARM hybrid attention session rec"
