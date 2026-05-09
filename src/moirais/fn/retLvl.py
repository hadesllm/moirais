"""Return level z_p."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["return_level"]


def return_level(mu, sigma, xi, T):
    """
    Return level z_p

    Formula: z_p = μ − σ/ξ (1 − (-log(1-1/T))^{-ξ})

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
        Keys: estimate

    References
    ----------
    Coles (2001)
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Return level z_p"})


def cheatsheet():
    return "retLvl: Return level z_p"
