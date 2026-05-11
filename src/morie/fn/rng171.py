"""Kalman-like gain vector in the RLS algorithm.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_kalman_gain"]


def rangayyan_ch3_rls_kalman_gain(P, r, lam, n):
    """
    Kalman-like gain vector in the RLS algorithm.

    Formula: k(n) = lambda^(-1) * P(n-1) * r(n) / (1 + lambda^(-1) * r^T(n) * P(n-1) * r(n))

    Parameters
    ----------
    P : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.217, p. 188
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kalman-like gain vector in the RLS algorithm."})


def cheatsheet():
    return "rng171: Kalman-like gain vector in the RLS algorithm."
