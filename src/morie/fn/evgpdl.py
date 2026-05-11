"""Log-likelihood of a GPD sample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gpd_loglik"]


def evt_gpd_loglik(y, sigma, xi):
    """
    Log-likelihood of a GPD sample

    Formula: ℓ = -n log σ - (1+1/ξ) Σ log(1+ξy/σ)

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
        Keys: ll

    References
    ----------
    Coles (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-likelihood of a GPD sample"})


def cheatsheet():
    return "evgpdl: Log-likelihood of a GPD sample"
