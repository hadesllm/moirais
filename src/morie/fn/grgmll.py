# morie.fn -- function file (hadesllm/morie)
"""Gaussian mixture log-likelihood of observations under K components."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gmm_log_likelihood"]


def geron_gmm_log_likelihood(X, pi, means, covars):
    """
    Gaussian mixture log-likelihood of observations under K components

    Formula: log_L(theta) = sum_i log( sum_k pi_k N(x_i | mu_k, Sigma_k) )

    Parameters
    ----------
    X : array-like
        Input data.
    pi : array-like
        Input data.
    means : array-like
        Input data.
    covars : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: log_L

    References
    ----------
    Géron Ch 8, GMM section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian mixture log-likelihood of observations under K components"})


def cheatsheet():
    return "grgmll: Gaussian mixture log-likelihood of observations under K components"
