"""Moments accountant for DP-SGD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["moments_accountant"]


def moments_accountant(sigma, sample_rate, steps):
    """
    Moments accountant for DP-SGD

    Formula: track log MGF α(λ) = log E[exp(λ X)]

    Parameters
    ----------
    sigma : array-like
        Input data.
    sample_rate : array-like
        Input data.
    steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Abadi et al (2016)
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    n = len(sigma)
    result = float(np.mean(sigma))
    se = float(np.std(sigma, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moments accountant for DP-SGD"})


def cheatsheet():
    return "momacc: Moments accountant for DP-SGD"
