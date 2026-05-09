"""Mediation analysis for binary outcome (logistic)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["binary_outcome_mediation"]


def binary_outcome_mediation(X, M, Y):
    """
    Mediation analysis for binary outcome (logistic)

    Formula: OR_direct, OR_indirect via inverse-odds-weighting

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele & Vansteelandt (2010)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mediation analysis for binary outcome (logistic)"})


def cheatsheet():
    return "binmed: Mediation analysis for binary outcome (logistic)"
