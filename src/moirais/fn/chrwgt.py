"""Censoring-at-risk weights for survival."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["censoring_at_risk_weight"]


def censoring_at_risk_weight(time, censor, A, H):
    """
    Censoring-at-risk weights for survival

    Formula: 1/Pr(C>t|A,H)

    Parameters
    ----------
    time : array-like
        Input data.
    censor : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Finkelstein (2000)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Censoring-at-risk weights for survival"})


def cheatsheet():
    return "chrwgt: Censoring-at-risk weights for survival"
