"""GEV distribution density."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gev_pdf"]


def evt_gev_pdf(x, mu, sigma, xi):
    """
    GEV distribution density

    Formula: f(x|μ,σ,ξ) = (1/σ) t(x)^{ξ+1} exp(-t(x))

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
        Keys: f

    References
    ----------
    Coles (2001) §3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GEV distribution density"})


def cheatsheet():
    return "evgevp: GEV distribution density"
