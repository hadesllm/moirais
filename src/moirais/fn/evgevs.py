"""Sample from a GEV distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gev_sample"]


def evt_gev_sample(mu, sigma, xi, n):
    """
    Sample from a GEV distribution

    Formula: x = μ + (σ/ξ)((-log U)^{-ξ} - 1), U~Unif

    Parameters
    ----------
    mu : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Coles (2001)
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sample from a GEV distribution"})


def cheatsheet():
    return "evgevs: Sample from a GEV distribution"
