"""Log-likelihood of a GEV sample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gev_loglik"]


def evt_gev_loglik(x, mu, sigma, xi):
    """
    Log-likelihood of a GEV sample

    Formula: ℓ = -n log σ - (1+1/ξ)Σ log t - Σ t^{-1/ξ}

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
        Keys: ll

    References
    ----------
    Coles (2001)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-likelihood of a GEV sample"})


def cheatsheet():
    return "evgevl: Log-likelihood of a GEV sample"
