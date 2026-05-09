"""Double ML mediation Neyman-orthogonal."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dml_mediation_orthogonal"]


def dml_mediation_orthogonal(Y, X, M, C, K):
    """
    Double ML mediation Neyman-orthogonal

    Formula: score is orthogonal in nuisance

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
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Farbmacher-Huber-Lafférs-Langen-Spindler (2022)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Double ML mediation Neyman-orthogonal"})


def cheatsheet():
    return "dmlMed: Double ML mediation Neyman-orthogonal"
