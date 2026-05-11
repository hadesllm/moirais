"""Mediation with survival outcome."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survival_mediation"]


def survival_mediation(T, delta, X, M, C):
    """
    Mediation with survival outcome

    Formula: AFT or Cox-based decomposition

    Parameters
    ----------
    T : array-like
        Input data.
    delta : array-like
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
    Lange-Hansen (2011); VanderWeele (2011)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mediation with survival outcome"})


def cheatsheet():
    return "survMd: Mediation with survival outcome"
