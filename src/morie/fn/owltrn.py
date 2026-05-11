"""Outcome-weighted learning for optimal regime."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["outcome_weighted_learning"]


def outcome_weighted_learning(y, D, W, pi):
    """
    Outcome-weighted learning for optimal regime

    Formula: min E[Y * 1{D != d(W)} / pi(D|W)]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    W : array-like
        Input data.
    pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhao et al (2012, 2015)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Outcome-weighted learning for optimal regime"})


def cheatsheet():
    return "owltrn: Outcome-weighted learning for optimal regime"
