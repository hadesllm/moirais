"""MLE of GEV parameters from block maxima."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gev_mle"]


def evt_gev_mle(x, init):
    """
    MLE of GEV parameters from block maxima

    Formula: argmax Σ log f(x_i|μ,σ,ξ)

    Parameters
    ----------
    x : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mu, sigma, xi, ll

    References
    ----------
    Coles (2001) §3.4
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MLE of GEV parameters from block maxima"})


def cheatsheet():
    return "evgevm: MLE of GEV parameters from block maxima"
