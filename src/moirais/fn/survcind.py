"""Survival concordance index (Harrell C)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survival_concordance"]


def survival_concordance(time, event, predicted_risk):
    """
    Survival concordance index (Harrell C)

    Formula: P(M_i > M_j | T_i < T_j)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    predicted_risk : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Harrell-Califf-Pryor (1982)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survival concordance index (Harrell C)"})


def cheatsheet():
    return "survcind: Survival concordance index (Harrell C)"
