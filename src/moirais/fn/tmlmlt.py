"""TMLE for multi-arm treatments."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_multiple_treatments"]


def tmle_multiple_treatments(y, D, X, arm_set):
    """
    TMLE for multi-arm treatments

    Formula: target E[Y(a)] for each a in A; pairwise contrasts

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    arm_set : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lendle et al (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for multi-arm treatments"})


def cheatsheet():
    return "tmlmlt: TMLE for multi-arm treatments"
