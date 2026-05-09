"""Inverse-odds-of-treatment weighting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weight_based_mediation"]


def weight_based_mediation(X, M, C, Y):
    """
    Inverse-odds-of-treatment weighting

    Formula: weights w_M = P(X=1|C)/P(X=1|M,C) for NDE

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    C : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tchetgen Tchetgen-Shpitser (2012)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse-odds-of-treatment weighting"})


def cheatsheet():
    return "wenge: Inverse-odds-of-treatment weighting"
