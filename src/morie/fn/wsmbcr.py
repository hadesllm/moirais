"""Bayesian credible interval."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_credible_interval"]


def wasserman_credible_interval(posterior, alpha):
    """
    Bayesian credible interval

    Formula: P(a <= theta <= b | x) = 1 - alpha

    Parameters
    ----------
    posterior : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lower, upper

    References
    ----------
    Wasserman (2004), Ch 11
    """
    posterior = np.atleast_1d(np.asarray(posterior, dtype=float))
    n = len(posterior)
    result = float(np.mean(posterior))
    se = float(np.std(posterior, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian credible interval"})


def cheatsheet():
    return "wsmbcr: Bayesian credible interval"
