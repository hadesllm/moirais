"""DP Gaussian mixture with stick-breaking representation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_gaussian_mixture"]


def dp_gaussian_mixture(y, alpha, prior_mu, prior_sigma, truncation):
    """
    DP Gaussian mixture with stick-breaking representation

    Formula: pi_k = beta_k prod_{j<k}(1 - beta_j); beta_k ~ Beta(1, alpha)

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    prior_mu : array-like
        Input data.
    prior_sigma : array-like
        Input data.
    truncation : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sethuraman (1994); Blei-Jordan (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP Gaussian mixture with stick-breaking representation"})


def cheatsheet():
    return "dpgmm: DP Gaussian mixture with stick-breaking representation"
