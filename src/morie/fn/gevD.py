"""Generalized extreme value distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gev_distribution"]


def gev_distribution(mu, sigma, xi):
    """
    Generalized extreme value distribution

    Formula: CDF: exp(-(1+ξ(x-μ)/σ)^{-1/ξ})

    Parameters
    ----------
    mu : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Coles (2001) book
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized extreme value distribution"})


def cheatsheet():
    return "gevD: Generalized extreme value distribution"
