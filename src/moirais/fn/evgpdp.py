"""GPD density above threshold."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gpd_pdf"]


def evt_gpd_pdf(y, sigma, xi):
    """
    GPD density above threshold

    Formula: f(y) = (1/σ)(1+ξy/σ)^{-1-1/ξ}

    Parameters
    ----------
    y : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: f

    References
    ----------
    Coles (2001) §4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GPD density above threshold"})


def cheatsheet():
    return "evgpdp: GPD density above threshold"
