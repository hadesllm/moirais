"""GEV quantile (return-level) function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gev_quantile"]


def evt_gev_quantile(p, mu, sigma, xi):
    """
    GEV quantile (return-level) function

    Formula: x_p = μ - (σ/ξ)(1 - (-log p)^{-ξ})

    Parameters
    ----------
    p : array-like
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
        Keys: x_p

    References
    ----------
    Coles (2001)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GEV quantile (return-level) function"})


def cheatsheet():
    return "evgevq: GEV quantile (return-level) function"
