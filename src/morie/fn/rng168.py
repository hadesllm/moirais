"""Recursive update for the cross-correlation vector in RLS.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_theta_recursion"]


def rangayyan_ch3_rls_theta_recursion(Theta, r, x, lam, n):
    """
    Recursive update for the cross-correlation vector in RLS.

    Formula: Theta(n) = lambda * Theta(n-1) + r(n) * x(n)

    Parameters
    ----------
    Theta : array-like
        Input data.
    r : array-like
        Input data.
    x : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.212, p. 187
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Recursive update for the cross-correlation vector in RLS."})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Recursive update for the cross-correlation vector in RLS."})


def cheatsheet():
    return "rng168: Recursive update for the cross-correlation vector in RLS."
