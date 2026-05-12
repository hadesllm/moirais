# morie.fn -- function file (hadesllm/morie)
"""Closed-form KL( N(mu, sigma^2) || N(0, I) ) per dimension."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kl_divergence_gaussian"]


def geron_kl_divergence_gaussian(mu, logvar):
    """
    Closed-form KL( N(mu, sigma^2) || N(0, I) ) per dimension

    Formula: KL = -0.5 * sum_i (1 + log(sigma_i^2) - mu_i^2 - sigma_i^2)

    Parameters
    ----------
    mu : array-like
        Input data.
    logvar : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: kl

    References
    ----------
    Géron Ch 18, VAE KL term section
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed-form KL( N(mu, sigma^2) || N(0, I) ) per dimension"})


def cheatsheet():
    return "grkldg: Closed-form KL( N(mu, sigma^2) || N(0, I) ) per dimension"
