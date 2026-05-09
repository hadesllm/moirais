"""STAMP short-term attention/memory rec."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stamp"]


def stamp(sessions, K):
    """
    STAMP short-term attention/memory rec

    Formula: attention to current click + session memory

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
    Liu et al (2018)
    """
    sessions = np.atleast_1d(np.asarray(sessions, dtype=float))
    n = len(sessions)
    result = float(np.mean(sessions))
    se = float(np.std(sessions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "STAMP short-term attention/memory rec"})


def cheatsheet():
    return "strec: STAMP short-term attention/memory rec"
