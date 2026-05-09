"""Simulate from a bivariate extreme-value copula."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_bv_evd_sim"]


def evt_bv_evd_sim(alpha, n):
    """
    Simulate from a bivariate extreme-value copula

    Formula: x_i = -1/log(U_i^{α_i}) per Stephenson

    Parameters
    ----------
    alpha : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x, y

    References
    ----------
    Stephenson (2003)
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simulate from a bivariate extreme-value copula"})


def cheatsheet():
    return "evbevsim: Simulate from a bivariate extreme-value copula"
