"""T-year return level under GEV."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_return_level"]


def evt_return_level(mu, sigma, xi, T):
    """
    T-year return level under GEV

    Formula: z_T = μ - (σ/ξ)(1 - (-log(1-1/T))^{-ξ})

    Parameters
    ----------
    mu : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z_T

    References
    ----------
    Coles (2001)
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T-year return level under GEV"})


def cheatsheet():
    return "evrl: T-year return level under GEV"
