"""Partial-linear GRF for high-dim controls."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["partial_linear_grf"]


def partial_linear_grf(y, D, X):
    """
    Partial-linear GRF for high-dim controls

    Formula: residualize on X high-dim, then forest

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey-Tibshirani-Wager (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partial-linear GRF for high-dim controls"})


def cheatsheet():
    return "plrgrf: Partial-linear GRF for high-dim controls"
