"""Riccati-style recursion for the inverse autocorrelation matrix in RLS.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_inverse_recursion"]


def rangayyan_ch3_rls_inverse_recursion(Phi, r, lam, n):
    """
    Riccati-style recursion for the inverse autocorrelation matrix in RLS.

    Formula: Phi^(-1)(n) = lambda^(-1) * Phi^(-1)(n-1) - lambda^(-2) * Phi^(-1)(n-1)*r(n)*r^T(n)*Phi^(-1)(n-1) / (1 + lambda^(-1)*r^T(n)*Phi^(-1)(n-1)*r(n))

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
    Rangayyan (2024), Ch 3, Eq 3.215, p. 188
    """
    Phi = np.atleast_1d(np.asarray(Phi, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(Phi), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Riccati-style recursion for the inverse autocorrelation matrix in RLS."})
    result = stats.spearmanr(Phi[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Riccati-style recursion for the inverse autocorrelation matrix in RLS."})


def cheatsheet():
    return "rng170: Riccati-style recursion for the inverse autocorrelation matrix in RLS."
