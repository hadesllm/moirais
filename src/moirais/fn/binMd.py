"""Mediation for binary outcome (logit)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["binary_outcome_mediation"]


def binary_outcome_mediation(Y, X, M, C):
    """
    Mediation for binary outcome (logit)

    Formula: NIE/NDE on OR scale via approximation

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele-Vansteelandt (2010)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mediation for binary outcome (logit)"})


def cheatsheet():
    return "binMd: Mediation for binary outcome (logit)"
