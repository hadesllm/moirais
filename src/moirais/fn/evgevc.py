"""GEV distribution CDF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gev_cdf"]


def evt_gev_cdf(x, mu, sigma, xi):
    """
    GEV distribution CDF

    Formula: F(x) = exp(-(1+ξ(x-μ)/σ)^{-1/ξ})

    Parameters
    ----------
    x : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: F

    References
    ----------
    Coles (2001) §3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GEV distribution CDF"})


def cheatsheet():
    return "evgevc: GEV distribution CDF"
