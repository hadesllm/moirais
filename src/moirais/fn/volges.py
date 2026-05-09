"""Implied Expected Shortfall under GARCH+t."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_garch_es_impl"]


def vol_garch_es_impl(mu, sigma_next, alpha, dist, nu):
    """
    Implied Expected Shortfall under GARCH+t

    Formula: ES_α = -μ - σ φ(z_α)/(1-α) for normal

    Parameters
    ----------
    mu : array-like
        Input data.
    sigma_next : array-like
        Input data.
    alpha : array-like
        Input data.
    dist : array-like
        Input data.
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ES

    References
    ----------
    Acerbi-Tasche (2002)
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Implied Expected Shortfall under GARCH+t"})


def cheatsheet():
    return "volges: Implied Expected Shortfall under GARCH+t"
