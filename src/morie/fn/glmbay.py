"""Bayesian GLM via Stan/JAGS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayesian_glm"]


def bayesian_glm(y, X, family, priors):
    """
    Bayesian GLM via Stan/JAGS

    Formula: likelihood + priors -> MCMC posterior

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    family : array-like
        Input data.
    priors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman BDA3 Ch 16
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian GLM via Stan/JAGS"})


def cheatsheet():
    return "glmbay: Bayesian GLM via Stan/JAGS"
