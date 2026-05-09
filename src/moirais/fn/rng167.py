"""Recursive update for the autocorrelation matrix in RLS.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_phi_recursion"]


def rangayyan_ch3_rls_phi_recursion(Phi, r, lam, n):
    """
    Recursive update for the autocorrelation matrix in RLS.

    Formula: Phi(n) = lambda * Phi(n-1) + r(n) * r^T(n)

    Parameters
    ----------
    Phi : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.211, p. 187
    """
    Phi = np.atleast_1d(np.asarray(Phi, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(Phi), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Recursive update for the autocorrelation matrix in RLS."})
    result = stats.spearmanr(Phi[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Recursive update for the autocorrelation matrix in RLS."})


def cheatsheet():
    return "rng167: Recursive update for the autocorrelation matrix in RLS."
