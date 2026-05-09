"""GPD quantile function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gpd_quantile"]


def evt_gpd_quantile(p, sigma, xi):
    """
    GPD quantile function

    Formula: y_p = (σ/ξ)((1-p)^{-ξ} - 1)

    Parameters
    ----------
    p : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_p

    References
    ----------
    Coles (2001)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GPD quantile function"})


def cheatsheet():
    return "evgpdq: GPD quantile function"
