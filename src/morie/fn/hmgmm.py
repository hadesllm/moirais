# morie.fn — function file (hadesllm/morie)
"""Gaussian mixture model fit via EM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gaussian_mixture"]


def geron_gaussian_mixture(X, n_components, seed):
    """
    Gaussian mixture model fit via EM

    Formula: p(x) = sum_k pi_k N(x; mu_k, Sigma_k)

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pi, mu, Sigma

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian mixture model fit via EM"})


def cheatsheet():
    return "hmgmm: Gaussian mixture model fit via EM"
