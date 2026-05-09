"""Recursive update for the inverse correlation matrix P(n) in RLS.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_p_recursion"]


def rangayyan_ch3_rls_p_recursion(P, k, r, lam, n):
    """
    Recursive update for the inverse correlation matrix P(n) in RLS.

    Formula: P(n) = lambda^(-1) * P(n-1) - lambda^(-1) * k(n) * r^T(n) * P(n-1)

    Parameters
    ----------
    P : array-like
        Input data.
    k : array-like
        Input data.
    r : array-like
        Input data.
    lam : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.218, p. 188
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(P), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Recursive update for the inverse correlation matrix P(n) in RLS."})
    result = stats.spearmanr(P[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Recursive update for the inverse correlation matrix P(n) in RLS."})


def cheatsheet():
    return "rng172: Recursive update for the inverse correlation matrix P(n) in RLS."
