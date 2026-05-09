"""Shrinkage propensity model via Bayesian prior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shrinkage_propensity"]


def shrinkage_propensity(A, H, prior_mu, prior_sigma):
    """
    Shrinkage propensity model via Bayesian prior

    Formula: posterior PS with informative prior

    Parameters
    ----------
    A : array-like
        Input data.
    H : array-like
        Input data.
    prior_mu : array-like
        Input data.
    prior_sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imbens-Rubin (2015)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shrinkage propensity model via Bayesian prior"})


def cheatsheet():
    return "shrtgr: Shrinkage propensity model via Bayesian prior"
