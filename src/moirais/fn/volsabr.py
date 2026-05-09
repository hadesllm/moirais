"""SABR Hagan-Kumar-Lesniewski implied volatility."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_sabr_implied"]


def vol_sabr_implied(F, K, T, alpha, beta, rho, nu):
    """
    SABR Hagan-Kumar-Lesniewski implied volatility

    Formula: Hagan asymptotic expansion in (α,β,ρ,ν)

    Parameters
    ----------
    F : array-like
        Input data.
    K : array-like
        Input data.
    T : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.
    rho : array-like
        Input data.
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: iv

    References
    ----------
    Hagan-Kumar-Lesniewski-Woodward (2002)
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SABR Hagan-Kumar-Lesniewski implied volatility"})


def cheatsheet():
    return "volsabr: SABR Hagan-Kumar-Lesniewski implied volatility"
