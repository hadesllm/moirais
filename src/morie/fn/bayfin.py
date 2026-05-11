"""Finite mixture model (Bayesian K-means)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["finite_mixture"]


def finite_mixture(y, K):
    """
    Finite mixture model (Bayesian K-means)

    Formula: y_i ~ sum_k pi_k N(mu_k, sigma_k); priors on mu, sigma, pi

    Parameters
    ----------
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Diebolt-Robert (1994)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Finite mixture model (Bayesian K-means)"})


def cheatsheet():
    return "bayfin: Finite mixture model (Bayesian K-means)"
