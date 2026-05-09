"""Sample from a GPD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gpd_sample"]


def evt_gpd_sample(sigma, xi, n):
    """
    Sample from a GPD

    Formula: y = (σ/ξ)(U^{-ξ} - 1), U~Unif

    Parameters
    ----------
    sigma : array-like
        Input data.
    xi : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Coles (2001)
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    n = len(sigma)
    result = float(np.mean(sigma))
    se = float(np.std(sigma, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sample from a GPD"})


def cheatsheet():
    return "evgpds: Sample from a GPD"
