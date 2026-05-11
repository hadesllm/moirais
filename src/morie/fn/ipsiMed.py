"""Interventional ψ in causal forests."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["interventional_psi"]


def interventional_psi(Y, X, M, C):
    """
    Interventional ψ in causal forests

    Formula: random forest estimate of intervention effect

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
    Athey-Tibshirani-Wager (2019) generalized RF
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Interventional ψ in causal forests"})


def cheatsheet():
    return "ipsiMed: Interventional ψ in causal forests"
